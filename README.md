# RARE 2025 Challenge -- Early Cancer Detection

## 📖 Overview

This repository contains my work for the **RARE 2025 Challenge**, part
of **EndoVis**. The focus is detecting **early-stage cancers** in
patients with **Barrett's Esophagus (BE)** during routine endoscopy.
These cases are rare (\<1% prevalence) and subtle, making detection
extremely challenging.

The project implements a **classification system** trained on endoscopic
images to detect early neoplasia in BE, aiming to build a model that
is: - Sensitive enough to catch early cancers - Specific enough to avoid
flooding clinicians with false positives

The system is containerized with **Docker** for easy deployment.

---

## 🎯 Project Goals

-   Develop a **robust classification model** for early-stage cancer in
    BE.
-   Train and evaluate models in a **low-prevalence setting**.
-   Provide a reproducible **benchmark pipeline**.
-   Package the model in a **Docker container** for portability and
    deployment.

---

## 🧑‍⚕️ The Clinical Problem

-   Early BE-associated cancers are **rare (\<1%)** and **difficult to
    detect**.
-   Subtle signs are often missed in real-world practice.
-   Early detection enables endoscopic treatment with \>90% long-term
    success.
-   If missed, progression leads to poor outcomes and \~15% five‑year
    survival rates.

---

## 📊 Dataset & Challenge Context

-   Based on the **RARE 2025 Challenge dataset**, hosted on Hugging
    Face.
-   Highly **imbalanced distribution**: normal/benign cases vastly
    outnumber early neoplasia.
-   Models must strike the **right balance between sensitivity &
    specificity**.

**Dataset Link:**\
[RARE25‑train on Hugging
Face](https://huggingface.co/datasets/TimJaspersTue/RARE25-train)\
Please note: You need to log in and accept contributor conditions to
access the data.

---

## 🏗️ Model Architecture  
The model was designed to balance **sensitivity** (catching rare cases) and **specificity** (avoiding false alarms). Key components include:  

- **Backbone:** Deep convolutional neural network (CNN) with transfer learning from an ImageNet-pretrained model.  
- **Feature Extractor:** Multi-scale feature maps to capture both subtle local anomalies and global patterns.  
- **Classifier Head:** Fully connected layers with dropout for regularization.  
- **Loss Function:** Weighted cross-entropy / focal loss to address extreme class imbalance.  
- **Optimization:** Adam optimizer with cyclical learning rate scheduling.  

---

## 🧪 Pre-processing & Data Handling  
To prepare the dataset and mitigate imbalance challenges:  

- **Image Normalization:** Standardized pixel values across samples.  
- **Resizing & Cropping:** Input frames resized to a consistent resolution suitable for the backbone network.  
- **Augmentation:** Applied rotation, flipping, brightness, and color jittering to improve generalization.  
- **Patch Extraction:** Focused sampling of regions-of-interest (ROI) to emphasize subtle lesions.  
- **Class Balancing:** Used weighted sampling and synthetic oversampling of minority class.  
- **Cross-validation:** Ensured robust performance estimation despite small positive class size.  

---

## 🏆 Challenge Progress

The RARE 2025 Challenge is ongoing:

-   **Sanity Check Phase:** 🥇 1st place (as of **06/09/2025**)

![sanity_check](./images/sanity_check.png)

[Sanity Check Leaderboard](https://rare25.grand-challenge.org/evaluation/test-submission-debug/leaderboard/)

-   **Open Development Phase:** 📊 36th place (as of **06/09/2025**)
![Open_Development](./images/open_development.png)
[Open Development Phase Leaderboard](https://rare25.grand-challenge.org/evaluation/open-development-phase/leaderboard/)
---

## 📦 Model Deployment

-   The trained model is packaged in a **Docker image**.\
-   Container exposes an API endpoint per your setup (e.g., REST).\
-   Supports integration into clinical pipelines for testing and
    evaluation.

---

## 📌 Note  
Not all project files are included in this repository at the moment.  
👉 Once the project is fully complete, I will upload **all files and resources** here for reproducibility and open development.  
