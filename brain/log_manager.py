"""
Centralized logging manager for SyzPilot-Brain.

Log structure:
    brain/logs/
    ├── controller.log              # Main controller events
    ├── guidance.log                # Guidance results (attribution scores, weights, templates)
    ├── model_versions.log          # Model version tracking
    └── {task_name}/{run_id}/       # Per-task subdirectory
        ├── receiver.log            # Data receiving, training triggers
        ├── trainer.log             # Training process output
        └── attributor.log          # Attribution analysis results
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional


class LogManager:
    """Manages all brain-side loggers with proper file handlers."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._loggers = {}

        # Create singleton loggers
        self.controller = self._create_logger("controller", self.log_dir / "controller.log")
        self.guidance = self._create_logger("guidance", self.log_dir / "guidance.log")
        self.model_versions = self._create_logger("model_versions", self.log_dir / "model_versions.log")

    def _create_logger(self, name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
        """Create a logger with file and stream handlers."""
        logger = logging.getLogger(f"syzpilot.{name}")
        logger.setLevel(level)
        logger.propagate = False

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # File handler
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)

        # Stream handler (for controller and guidance only)
        if name in ("controller", "guidance"):
            sh = logging.StreamHandler()
            sh.setLevel(level)
            sh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            ))
            logger.addHandler(sh)

        return logger

    def get_task_logger(self, task_name: str, run_id: int, component: str) -> logging.Logger:
        """Get or create a per-task logger for receiver/trainer/attributor.

        Args:
            task_name: Task name (e.g., "kernel BUG in validate_xmit_skb")
            run_id: Run ID
            component: "receiver", "trainer", or "attributor"

        Returns:
            Logger writing to brain/logs/{task_name}/{run_id}/{component}.log
        """
        safe_name = task_name.replace(" ", "_").replace("/", "_")
        key = f"{safe_name}/{run_id}/{component}"

        with self._lock:
            if key in self._loggers:
                return self._loggers[key]

            task_log_dir = self.log_dir / safe_name / str(run_id)
            task_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = task_log_dir / f"{component}.log"

            logger = self._create_logger(key, log_file)
            self._loggers[key] = logger
            return logger

    def log_guidance_result(self, task_id: str, version: int,
                            static_weights: dict, attribution_weights: dict,
                            merged_weights: dict, templates: list,
                            send_success: bool):
        """Log a complete guidance result to guidance.log.

        Args:
            task_id: Task identifier
            version: Guidance version number
            static_weights: Weights from PathBasedAnalyzer
            attribution_weights: Weights from Captum IG
            merged_weights: Final merged weights
            templates: Mutation templates
            send_success: Whether guidance was sent successfully
        """
        # Build entire message as single string to avoid interleaving from concurrent tasks
        lines = [f"{'=' * 60}"]
        lines.append(f"Task: {task_id}  Version: {version}  Sent: {send_success}")

        # Static analysis weights
        if static_weights:
            top_static = sorted(static_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append(f"  Static ({len(static_weights)} syscalls): "
                         f"{', '.join(f'{s}={w:.3f}' for s, w in top_static)}")
        else:
            lines.append(f"  Static: (none)")

        # Attribution weights
        if attribution_weights:
            top_attr = sorted(attribution_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append(f"  Attribution ({len(attribution_weights)} syscalls): "
                         f"{', '.join(f'{s}={w:.3f}' for s, w in top_attr)}")
        else:
            lines.append(f"  Attribution: (none, insufficient positive samples)")

        # Merged weights
        top_merged = sorted(merged_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append(f"  Merged ({len(merged_weights)} syscalls): "
                     f"{', '.join(f'{s}={w:.3f}' for s, w in top_merged)}")

        # Templates
        if templates:
            lines.append(f"  Templates ({len(templates)}):")
            for i, tmpl in enumerate(templates[:5]):
                arg_hints = tmpl.get("arg_hints", [])
                lines.append(f"    [{i}] type={tmpl.get('type')} "
                             f"syscalls={tmpl.get('syscalls')} "
                             f"priority={tmpl.get('priority', 0):.3f} "
                             f"arg_hints={len(arg_hints)}")
        else:
            lines.append(f"  Templates: (none)")

        self.guidance.info("\n".join(lines))

    def log_model_version(self, task_id: str, model_name: str, model_version: str,
                          action: str, success: bool):
        """Log model version changes.

        Args:
            task_id: Task identifier
            model_name: TorchServe model name
            model_version: Model version string
            action: "deploy", "register", "deregister", "notify"
            success: Whether the action succeeded
        """
        status = "OK" if success else "FAIL"
        self.model_versions.info(f"[{status}] task={task_id} model={model_name} "
                                 f"version={model_version} action={action}")


# Global instance (initialized by controller)
_log_manager: Optional[LogManager] = None


def init_log_manager(log_dir: str = "./logs") -> LogManager:
    """Initialize the global log manager."""
    global _log_manager
    _log_manager = LogManager(log_dir)
    return _log_manager


def get_log_manager() -> LogManager:
    """Get the global log manager. Must call init_log_manager first."""
    if _log_manager is None:
        raise RuntimeError("LogManager not initialized. Call init_log_manager() first.")
    return _log_manager
