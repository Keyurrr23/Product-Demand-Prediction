# Product Demand Prediction — MLOps Practical Project

**Model:** Random Forest Regressor  
**Application:** Flask UI + JSON REST API  
**Packaging:** Docker  
**Deployment path:** AWS EC2 and Google Cloud Compute Engine

> **Important data note:** If `data/demand_data.csv` is absent, the training script creates a synthetic demonstration dataset. It proves the pipeline works but is not suitable for inventory decisions. Replace it with historical sales/demand data before claiming business accuracy.

## 1. Requirements
- Python 3.10 or 3.11
- VS Code or another Python IDE
- Docker Desktop for the container practical
- An AWS or Google Cloud account only for the relevant deployment practical

## 2. Setup (Windows PowerShell)
```powershell
mkdir product-demand-prediction
# Extract this project into that folder, then:
cd product_demand_prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## 3. Train and serialize model
```powershell
python train_model.py
```
This creates `data/demand_data.csv` (demo data only, if no file existed) and `model.pkl`, and prints MAE, RMSE and R². Do not report synthetic-data metrics as real-world results.

## 4. Run Flask application
```powershell
python app.py
```
Open http://127.0.0.1:5000. Submit the form to estimate demand.
- `GET /` — web form
- `GET /health` — service health
- `GET /predict` and `POST /predict` — browser form
- `POST /api/predict` — JSON prediction API

Example API request:
```powershell
curl.exe -X POST http://127.0.0.1:5000/api/predict `
  -H "Content-Type: application/json" `
  -d "{\"product_id\":\"P001\",\"category\":\"Grocery\",\"price\":80,\"promotion\":1,\"stock_available\":100,\"day_of_week\":5,\"month\":10}"
```
Example health check:
```powershell
curl.exe http://127.0.0.1:5000/health
```

## 5. Test
```powershell
pytest -v
```
Tests cover the home page, health endpoint, a successful API prediction, and input validation.

## 6. Docker (Practical 2)
Make sure `model.pkl` exists, then:
```powershell
docker --version
docker build -t product-demand-prediction:1.0 .
docker images
docker run -d --name product-demand-app -p 5000:5000 product-demand-prediction:1.0
docker ps
docker logs product-demand-app
curl.exe http://localhost:5000/health
```
Open http://localhost:5000 and test a prediction.
Stop/remove:
```powershell
docker stop product-demand-app
docker rm product-demand-app
```

## 7. AWS EC2 deployment (Practical 3)
1. In AWS Console, select a region (for example Mumbai `ap-south-1`) and launch an Ubuntu LTS EC2 instance.
2. Create/select a key pair and store the private key securely.
3. In the security group, allow SSH port 22 from **your IP only**. For an app demo, allow TCP 5000 only from a trusted client IP where possible; avoid unrestricted public access.
4. SSH from your computer:
   ```bash
   ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
   ```
5. On EC2, install Docker:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl enable --now docker
   sudo docker --version
   ```
6. Transfer the project folder from your local machine (run locally; update paths):
   ```bash
   scp -i your-key.pem -r product_demand_prediction ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/
   ```
7. In the SSH terminal:
   ```bash
   cd /home/ubuntu/product_demand_prediction
   sudo docker build -t product-demand-prediction:1.0 .
   sudo docker run -d --restart unless-stopped --name product-demand-app -p 5000:5000 product-demand-prediction:1.0
   sudo docker ps
   sudo docker logs product-demand-app
   ```
8. Visit `http://YOUR_EC2_PUBLIC_IP:5000/health` and the home page. Replace the placeholder with the actual public IP.

## 8. Google Cloud deployment (Practical 4)
1. Create/select a Google Cloud project and configure billing if required.
2. Enable Compute Engine API; create a small Ubuntu LTS VM.
3. Add a VPC ingress firewall rule for TCP 5000 targeting this VM. Limit source IPs when possible. Use browser SSH to connect.
4. Install Docker:
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl enable --now docker
   ```
5. Transfer the project folder using `gcloud compute scp` from your local terminal (update VM name, zone and project):
   ```bash
   gcloud compute scp --recurse ./product_demand_prediction VM_NAME:~/product_demand_prediction --zone=YOUR_VM_ZONE --project=YOUR_PROJECT_ID
   ```
6. In the VM's SSH terminal:
   ```bash
   cd ~/product_demand_prediction
   sudo docker build -t product-demand-prediction:1.0 .
   sudo docker run -d --restart unless-stopped --name product-demand-app -p 5000:5000 product-demand-prediction:1.0
   sudo docker ps
   sudo docker logs product-demand-app
   ```
7. Open `http://YOUR_GCP_EXTERNAL_IP:5000/health` and test the form.

## 9. Dataset schema for real data
Replace `data/demand_data.csv` with historical records containing these columns:
- `product_id`: SKU or product identifier
- `category`: product category
- `price`: selling price
- `promotion`: 0 or 1
- `stock_available`: inventory available at the time of prediction
- `day_of_week`: Monday=0 through Sunday=6
- `month`: 1 through 12
- `demand_quantity`: actual units sold/demanded (target)

Keep one row per product/time period. Use only information available at prediction time; avoid leakage from future sales or post-period fields. For serious forecasting, use a time-based train/test split and compare against simple baselines.

## 10. Screenshots to collect for practical file
1. Virtual environment and dependencies installed.
2. Training metrics and `model.pkl` created.
3. Project tree in VS Code.
4. Flask web form and prediction result.
5. `/health` JSON and `/api/predict` response.
6. `pytest -v` passing.
7. Docker image build, `docker images`, running `docker ps`, container logs.
8. AWS EC2 instance/security group, SSH terminal, app via public IP.
9. Google Cloud VM/firewall, SSH terminal, app via external IP.

## 11. Limitations and next improvements
- Synthetic data is only a runnable demonstration.
- Model prediction is an estimate, not a guaranteed sales figure.
- Add time-based validation, baseline comparison, model versioning, monitoring, and authenticated HTTPS before production use.
 
 ## 12. AWS EC2 URL:- http://43.204.219.107:5000/
 ## 13. GCP URL:- http://34.93.73.94:5000/