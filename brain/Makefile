.PHONY: all collectorserver clean help

# default target: generate all protobuf code
all: collectorserver

# generate Python protobuf code for collector server on SyzPilot-brain side
collectorserver:
	@echo "Generating Python protobuf code for collector server (aka data receiver)..."
	python -m grpc_tools.protoc -I. \
		--python_out=. \
		--grpc_python_out=. \
		transmission.proto
	@echo "[+] Python code generated in ."

# clean generated files
clean:
	rm -rf transmission_pb2.py transmission_pb2_grpc.py

# help message
help:
	@echo "Available targets:"
	@echo "  all        	  - Generate all protobuf code"
	@echo "  collectorserver - Generate Python protobuf code for collector server (aka data receiver)"
	@echo "  clean   	  - Remove generated code"