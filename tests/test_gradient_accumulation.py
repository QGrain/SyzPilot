"""Regression test for the pinned Accelerate accumulation semantics."""

import unittest

import torch
from accelerate import Accelerator


class CountingSGD(torch.optim.SGD):
    """Count physical optimizer updates beneath AcceleratedOptimizer."""

    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.physical_steps = 0

    def step(self, closure=None):
        self.physical_steps += 1
        return super().step(closure)


class GradientAccumulationTest(unittest.TestCase):
    def test_four_micro_batches_produce_two_optimizer_updates(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = CountingSGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        model, wrapped_optimizer, wrapped_scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )

        logical_steps = 0
        for _ in range(4):
            with accelerator.accumulate(model):
                loss = model(torch.ones((1, 1))).sum()
                accelerator.backward(loss)
                wrapped_optimizer.step()
                wrapped_scheduler.step()
                wrapped_optimizer.zero_grad()
            if accelerator.sync_gradients:
                logical_steps += 1

        self.assertEqual(logical_steps, 2)
        self.assertEqual(optimizer.physical_steps, 2)
        self.assertEqual(scheduler.last_epoch, 2)
        accelerator.end_training()


if __name__ == "__main__":
    unittest.main()
