# Llama2 in Kubernetes with Gradio

## Prerequisites

* Deploy [Kind cluster with GPU](https://www.substratus.ai/blog/kind-with-gpus)

* Deploy Nginx Ingress Controller:

```md
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/deploy/static/provider/kind/deploy.yaml
```

* Deploy Nvidia GPU Operator

```md
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false
```

* Deploy Pod to Check nvidia-smi
```md
kubectl apply -f - << EOF
apiVersion: v1
kind: Pod
metadata:
  name: cuda-vectoradd
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda-vectoradd
    image: "nvcr.io/nvidia/k8s/cuda-sample:vectoradd-cuda11.7.1-ubuntu20.04"
    resources:
      limits:
        nvidia.com/gpu: 1
EOF
docker exec -ti k8s-control-plane ln -s /sbin/ldconfig /sbin/ldconfig.real
kubectl delete --all pod -n gpu-operator
```

## Deploy Llama2 in Kubernetes

* Deploy Llama2 in Kubernetes

```md
kubectl apply -k manifests/overlays/
```

