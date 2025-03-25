# Unsupervised Learning with Diffusion Maps and Orthogonal Latent Space Autoencoders for Wearable Physiological Signal Analysis: Disentangling Distribution and Individual Differences

## Project Overview

This project investigates the distribution and individual differences in cardiovascular pulse waveforms collected from a proprietary wearable device developed by FLOWEHEALTH INC. We employ unsupervised learning, with a primary focus on Diffusion Maps for manifold learning, and a custom Autoencoder architecture designed to learn an orthogonal latent space. Our goal is to disentangle the effects of variations arising from the wearable device itself and genuine physiological differences between individuals. By analyzing these unlabeled pulse waveforms, we aim to understand their overall distribution and identify distinct factors contributing to individual physiological profiles.

## Methodology

Our approach combines unsupervised learning with a specific manifold learning technique (Diffusion Maps) and a tailored Autoencoder architecture for disentangled representation learning:

1.  **Data Preprocessing:**
    * Collect raw cardiovascular pulse waveform data, where each data point represents a single pulse measured at the radial artery using a proprietary sensor and wearable mechanism from FLOWEHEALTH INC.
    * Perform necessary data cleaning, including handling missing values and removing noise.
    * Standardize or normalize the pulse waveform data to ensure comparability across different individuals or recordings.

2.  **Disentangled Representation Learning with Orthogonal Latent Space Autoencoders:**
    * We will utilize a custom Autoencoder architecture designed to learn an orthogonal latent space. This design aims to separate the latent representation into distinct, uncorrelated components that primarily capture either variations due to the FLOWEHEALTH INC. wearable device and its sensor or genuine physiological differences between individuals.
    * The architecture will be constrained by measured data, ensuring that the learned representation is grounded in the actual physiological signals. This constraint will help in distinguishing between technical variations and biological variations.
    * The encoder part of the Autoencoder will learn a compressed, orthogonal representation of the input pulse waveforms. The orthogonality constraint encourages the latent dimensions to be independent, facilitating the separation of device-related and physiology-related information.
    * Our goal is to achieve dimensionality reduction while preserving the key characteristics of the pulse waveform and explicitly separating the sources of variation in the latent space.

3.  **Manifold Learning with Diffusion Maps:**
    * Following the dimensionality reduction with the orthogonal latent space Autoencoder, we will primarily use **Diffusion Maps** for manifold learning.
    * Diffusion Maps is a non-linear dimensionality reduction technique based on the concept of diffusion processes on a graph constructed from the data. It is particularly effective at revealing the underlying geometric structure of the data and handling non-linear relationships.
    * By applying Diffusion Maps to the latent space learned by the Autoencoder, we aim to visualize the distribution of pulse waveforms in a lower-dimensional space and identify clusters or gradients that correspond to either device variations or distinct physiological characteristics.

4.  **Distribution Analysis and Identification of Differences:**
    * We will analyze the distribution of the learned latent representations (both before and after applying Diffusion Maps) in the low-dimensional space.
    * We will examine how individual pulse waveforms from different individuals are positioned and clustered, seeking to identify patterns related to both physiological uniqueness and potential device-induced variations.
    * By leveraging the orthogonal nature of the latent space, we will attempt to interpret the dimensions and understand which aspects of the pulse waveform are most influenced by physiological factors and which are more related to the wearable device.

## Data

This research uses proprietary cardiovascular pulse waveform data collected from a patented wearable device and sensor developed by FLOWEHEALTH INC. Each data point represents a single pulse waveform measured at the radial artery. Due to the sensitive and proprietary nature of this data, including its content and key experimental results, it will not be publicly disclosed in this repository and remains the intellectual property of FLOWEHEALTH INC.

## Expected Outcomes

This project anticipates achieving the following outcomes:

* **Visualization of Pulse Waveform Distribution using Diffusion Maps:** Presenting the distribution of pulse waveforms in a low-dimensional space obtained through the orthogonal latent space Autoencoder followed by Diffusion Maps.
* **Identification of Clusters and Gradients Reflecting Physiological and Device Variations:** Revealing patterns in the reduced-dimensional space that correlate with both individual physiological traits and potential variations introduced by the FLOWEHEALTH INC. wearable device and sensor.
* **Understanding the Distinct Influences of Physiology and Wearable Device:** Gaining insights into how underlying physiological differences and the characteristics of the wearable device independently affect the measured pulse waveforms, as separated in the orthogonal latent space.
* **Characterization of Individual Physiological Signatures:** Identifying distinct regions or patterns in the low-dimensional space that represent unique physiological signatures of different individuals.

## Tools and Technologies

* **Python:** Primary programming language.
* **Scipy:** For signal data processing and analysis.
* **NumPy:** For numerical computation.
* **Scikit-learn:** May be used for some preprocessing steps.
* **`diffmap` (or similar library):** For implementing Diffusion Maps.
* **PyTorch:** For building and training the orthogonal latent space Autoencoder model.
* * **PyQt5:** For input data visualization.
* **Matplotlib and Seaborn:** For output data visualization, particularly for visualizing the latent space and the results of Diffusion Maps.

## Future Work

Future research may involve:

* Further refining the orthogonal latent space Autoencoder architecture and training process.
* Exploring the temporal evolution of the disentangled representations.
* Investigating the correlation between the identified clusters/gradients and specific physiological conditions or individual characteristics (where such information is available internally).
* Developing methods to normalize for device-specific variations to obtain a clearer signal of underlying physiology.

## Contact Information

James Lin - AI/ML Algorithm Researcher
jameslin@flowehealth.com
