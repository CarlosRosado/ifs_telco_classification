# Variables
DOCKER_USERNAME = carlosrosado
PROJECT_NAME = ifs_ml
SERVING_IMAGE = serving-ifs-image
TRAINING_IMAGE = training-ifs-image
TAG = latest
K8S_DIR = ./kubernetes

# Default target
.PHONY: all
all: build push deploy-train wait-for-training wait-before-serving deploy-serving

# Build Docker images
.PHONY: build
build:
	@echo "Building Docker images..."
	docker build -t $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG) -f docker_serving/Dockerfile .
	docker build -t $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG) -f docker_training/Dockerfile .

# Push Docker images to the registry
.PHONY: push
push:
	@echo "Pushing Docker images to the registry..."
	docker push $(DOCKER_USERNAME)/$(SERVING_IMAGE):$(TAG)
	docker push $(DOCKER_USERNAME)/$(TRAINING_IMAGE):$(TAG)

# Deploy training to Kubernetes
.PHONY: deploy-train
deploy-train:
	@echo "Deploying training resources to Kubernetes..."
	kubectl apply -f $(K8S_DIR)/configmap.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-deployment.yaml
	kubectl apply -f $(K8S_DIR)/mlflow-service.yaml
	kubectl apply -f $(K8S_DIR)/training-deployment.yaml
	kubectl apply -f $(K8S_DIR)/pvc.yaml

# Wait for the training job to complete
.PHONY: wait-for-training
wait-for-training:
	@echo "Waiting for training to complete..."
    kubectl wait --for=condition=complete --timeout=1200s job/training-job || (echo "Training job failed or timed out!" && exit 1)

# Add a waiting time before deploying serving
.PHONY: wait-before-serving
wait-before-serving:
	@echo "Waiting for resources to stabilize before deploying serving..."
	sleep 30  # Wait for 30 seconds

# Deploy serving to Kubernetes
.PHONY: deploy-serving
deploy-serving:
	@echo "Deploying serving resources to Kubernetes..."
	kubectl apply -f $(K8S_DIR)/serving-deployment.yaml
	kubectl apply -f $(K8S_DIR)/serving-service.yaml

# Clean up Kubernetes resources
.PHONY: clean
clean:
	@echo "Cleaning up all Kubernetes resources..."
	kubectl delete -f $(K8S_DIR)/configmap.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/mlflow-service.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/serving-service.yaml || true
	kubectl delete -f $(K8S_DIR)/training-deployment.yaml || true
	kubectl delete -f $(K8S_DIR)/pvc.yaml || true
	kubectl delete job ifs-training-job || true

.PHONY: test
test:
	@echo "Running tests..."
	pytest tests --disable-warnings -q
	@echo "Tests completed."

# Rebuild and redeploy everything
.PHONY: rebuild
rebuild: clean all
	@echo "Rebuilt and redeployed everything."






