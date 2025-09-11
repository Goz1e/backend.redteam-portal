challenges = [
    {
      "id": "abs_v2",
      "name": "Auto Browser Sniffer v2",
      "category": "Detection & Security",
      "weight": 0.35,
      "status": "Active",
      "submissions": 127,
      "avgScore": 0.73,
      "timeRemaining": "14 days",
      "description": "Develop algorithms to detect automated browser activity and bot behavior patterns.",
      "template": "ab_sniffer_v2",
      "testingGuide": "testing_manual.md"
    },
    {
      "id": "neural_net",
      "name": "Neural Network Models",
      "category": "Machine Learning",
      "weight": 0.25,
      "status": "Active",
      "submissions": 89,
      "avgScore": 0.68,
      "timeRemaining": "8 days",
      "description": "Build and optimize neural network architectures for complex pattern recognition.",
      "template": "neural_models_v1",
      "testingGuide": "neural_testing.md"
    },
    {
      "id": "data_analysis",
      "name": "Advanced Data Analysis",
      "category": "Data Science",
      "weight": 0.2,
      "status": "Active",
      "submissions": 156,
      "avgScore": 0.81,
      "timeRemaining": "21 days",
      "description": "Create sophisticated data analysis pipelines for large-scale datasets.",
      "template": "data_analysis_v3",
      "testingGuide": "data_testing.md"
    },
    {
      "id": "blockchain_dev",
      "name": "Blockchain Development",
      "category": "Distributed Systems",
      "weight": 0.2,
      "status": "Upcoming",
      "submissions": 0,
      "avgScore": 0,
      "timeRemaining": "5 days to start",
      "description": "Develop decentralized applications and smart contract solutions.",
      "template": "blockchain_v1",
      "testingGuide": "blockchain_testing.md"
    }
]

miners = [
    {
      "id": "miner_user_001",
      "user_id": "user_001",
      "walletAddress": "0x5f8b2c8f9a3d5e7f1b8c2d4e6f8a1b3c5d7e9f4c1d",
      "totalScore": 0.89,
      "rank": 1,
      "submissions": 47,
      "successRate": 94,
      "totalEarned": "12.45 TAO",
      "joinDate": "2024-09-15T10:00:00",
      "lastActive": "2024-12-10T14:30:00",
      "trustTier": "gold",
      "publicProfile": True
    },
    {
      "id": "miner_user_002",
      "user_id": "user_002",
      "walletAddress": "0x9c4d1a7b2e5f8c3a6d9b4e7c0f3a6d9b2e5f8c1a2e8a",
      "totalScore": 0.84,
      "rank": 2,
      "submissions": 52,
      "successRate": 88,
      "totalEarned": "10.23 TAO",
      "joinDate": "2024-08-22T09:15:00",
      "lastActive": "2024-12-09T16:45:00",
      "trustTier": "silver",
      "publicProfile": True
    }
]

payout_history = [
    {
      "id": "batch_1234",
      "date": "Jan 20, 2025",
      "shard": "A",
      "amount": "0.45 TAO",
      "usd": "$22.73",
      "method": "TAO Direct",
      "status": "Completed"
    },
    {
      "id": "batch_1235",
      "date": "Jan 18, 2025",
      "shard": "B",
      "amount": "0.32 TAO",
      "usd": "$16.15",
      "method": "USDC Pool",
      "status": "Completed"
    },
    {
      "id": "batch_1236",
      "date": "Jan 15, 2025",
      "shard": "A",
      "amount": "0.67 TAO",
      "usd": "$33.82",
      "method": "TAO Direct",
      "status": "Processing"
    },
    {
      "id": "batch_1237",
      "date": "Jan 12, 2025",
      "shard": "C",
      "amount": "0.89 TAO",
      "usd": "$44.93",
      "method": "USDT Pool",
      "status": "Completed"
    }
]

shard_earnings = [
    {
      "shard": "A",
      "category": "ML & AI",
      "amount": "1.87 TAO",
      "percentage": 45
    },
    {
      "shard": "B",
      "category": "Data Analysis",
      "amount": "1.34 TAO",
      "percentage": 32.5
    },
    {
      "shard": "C",
      "category": "Blockchain Dev",
      "amount": "0.91 TAO",
      "percentage": 22.5
    }
]

recent_submissions = [
    {
      "challenge": "Neural Networks",
      "score": 0.91,
      "time": "2h ago",
      "status": "Processing"
    },
    {
      "challenge": "Data Analysis",
      "score": 0.87,
      "time": "1d ago",
      "status": "Completed"
    },
    {
      "challenge": "Browser Sniffer",
      "score": 0.76,
      "time": "2d ago",
      "status": "Completed"
    }
]

# Submissions data matching the new schema
submissions = [
    {
      "id": "sub_001",
      "miner": "miner_user_001",  # References miner profile ID
      "challenge": "abs_v2",     # References challenge ID
      "challenge_name": "Auto Browser Sniffer v2",
      "code": """
function detectBotBehavior(userAgent, clickPattern, mouseMovements) {
    // Advanced bot detection algorithm
    const suspiciousPatterns = [
        /HeadlessChrome/i,
        /PhantomJS/i,
        /Selenium/i
    ];
    
    let score = 0;
    
    // Check user agent
    if (suspiciousPatterns.some(pattern => pattern.test(userAgent))) {
        score += 0.5;
    }
    
    // Analyze click patterns
    if (clickPattern.variance < 0.1) {
        score += 0.3;
    }
    
    // Check mouse movements
    if (mouseMovements.length === 0) {
        score += 0.4;
    }
    
    return score > 0.7;
}
      """.strip(),
      "score": 0.92,
      "time": "2025-01-20T14:30:00",
      "status": "completed"
    },
    {
      "id": "sub_002", 
      "miner": "miner_user_001",
      "challenge": "neural_net",
      "challenge_name": "Neural Network Models",
      "code": """
import tensorflow as tf
import numpy as np

class AdvancedNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_layers[0], 
                                           activation='relu', 
                                           input_shape=(input_size,)))
        
        for layer_size in hidden_layers[1:]:
            self.model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(0.2))
        
        self.model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
        
    def compile_model(self):
        self.model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
    def train(self, X_train, y_train, epochs=100):
        return self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
      """.strip(),
      "score": 0.87,
      "time": "2025-01-19T10:15:00",
      "status": "completed"
    },
    {
      "id": "sub_003",
      "miner": "miner_user_002", 
      "challenge": "data_analysis",
      "challenge_name": "Advanced Data Analysis",
      "code": """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class DataAnalysisPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusterer = KMeans(n_clusters=5)
        
    def preprocess_data(self, df):
        # Handle missing values
        df_cleaned = df.fillna(df.median())
        
        # Remove outliers using IQR method
        Q1 = df_cleaned.quantile(0.25)
        Q3 = df_cleaned.quantile(0.75)
        IQR = Q3 - Q1
        
        df_filtered = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | 
                                  (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return df_filtered
    
    def analyze_patterns(self, data):
        scaled_data = self.scaler.fit_transform(data)
        clusters = self.clusterer.fit_predict(scaled_data)
        
        return {
            'clusters': clusters,
            'centroids': self.clusterer.cluster_centers_,
            'inertia': self.clusterer.inertia_
        }
      """.strip(),
      "score": 0.95,
      "time": "2025-01-20T09:45:00",
      "status": "completed"
    },
    {
      "id": "sub_004",
      "miner": "miner_user_002",
      "challenge": "neural_net",
      "challenge_name": "Neural Network Models",
      "code": """
import torch
import torch.nn as nn
import torch.optim as optim

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutionalNeuralNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
      """.strip(),
      "score": 0.78,
      "time": "2025-01-17T16:20:00",
      "status": "completed"
    }
]
