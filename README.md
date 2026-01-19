# Hybrid Text Classification System for AI vs Human Content Detection

This project presents a production-grade hybrid text classification system designed to distinguish between human-written and AI-generated text in modern web applications. The system leverages a carefully engineered combination of explainable algorithmic intelligence and transformer-based deep learning models to deliver high accuracy, transparency, and real-time performance.

The algorithmic classification pipeline is built using advanced Natural Language Processing (NLP) techniques and statistical analysis. It extracts and processes seven core linguistic features—including lexical diversity, sentence length variance, n-gram repetition patterns, punctuation density, contraction usage, stopword distribution, and readability metrics (Flesch Reading Ease). These features undergo normalization, weighted aggregation, and sigmoid-based scoring to generate an interpretable AI-likelihood score, enabling feature-level explainability and transparent decision-making.

In parallel, the system integrates a transformer-based deep learning model fine-tuned using the Hugging Face ecosystem. The model is trained on curated and labeled datasets of human-written and AI-generated text, employing modern tokenization, contextual embeddings, and probabilistic inference to capture deeper semantic and stylistic patterns beyond rule-based detection.

The complete solution is deployed as a scalable full-stack web application. The backend is developed using Python and Flask, exposing RESTful APIs for text preprocessing, model inference, and hybrid score fusion. The frontend is built with React, HTML5, CSS3, and JavaScript, providing a responsive and intuitive user interface for real-time text analysis and visualization of results. The hybrid decision engine seamlessly combines outputs from both pipelines, delivering model-wise confidence scores, explainability feedback, and reliable classification results.

Overall, the system demonstrates a robust fusion of traditional NLP, statistical modeling, modern deep learning, and full-stack engineering, making it suitable for applications such as academic integrity verification, content authenticity analysis, and large-scale AI-generated text detection.

