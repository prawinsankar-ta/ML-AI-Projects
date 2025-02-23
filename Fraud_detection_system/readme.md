# Fraud Detection Using PaySim Dataset

## Overview
This project focuses on detecting fraudulent transactions using insights from the PaySim dataset. The workflow involves data ingestion, preprocessing, model training, deployment, and monitoring using AWS services.

## Steps Involved

1. **Ingest and Preprocess Data**  
   - Load the PaySim dataset.
   - Clean and preprocess the transaction data for model training.

2. **Train and Evaluate the Fraud Detection Model**  
   - Implement machine learning algorithms for fraud detection.
   - Evaluate the model's performance using appropriate metrics.

3. **Deploy the Model on AWS**  
   - Develop a scalable API for fraud detection using AWS services.
   
4. **Monitor Fraud Detection Results**  
   - Utilize AWS analytics services to track and improve model performance.

## Deployment
- The model is deployed as an API on AWS, ensuring scalability and real-time fraud detection.

## Monitoring
- AWS analytics services, such as CloudWatch and S3 logging, are used to monitor API performance and fraud detection trends.

## Visualization
![Fraud Detection Architecture](solution/solution.png)

## Requirements
- Python 3.x
- AWS SDK (Boto3)
- Scikit-learn, Pandas, NumPy
- Flask/FastAPI for API deployment
- AWS services: S3, Lambda, API Gateway, CloudWatch

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fraud-detection.git
   cd fraud-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Deploy to AWS:
   ```bash
   sh deploy.sh
   ```

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.

