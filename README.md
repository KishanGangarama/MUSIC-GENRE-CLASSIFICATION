# MUSIC-GENRE-CLASSIFICATION
# Abstract
By creating a trustworthy and effective system for automatic classification utilizing machine learning algorithms and audio attributes taken from a dataset of 8,000 recordings, this project seeks to address the issue of manual classification of music genres. Moreover, clustering strategies will be used to find commonalities among overlapping genres. The outcomes of this project could be applied to music-related applications including playlist makers, recommendation systems, and music search engines.

# I.	INTRODUCTION
It is a challenging task that might have a big impact on many music-related applications to automatically classify musical genres and identify similarities between them. The present manual classification techniques take a lot of time and are subjective, which causes genre classification to be inconsistent. In order to solve this issue, the project’s goal is to create an accurate and effective system for automatically classifying genres using machine learning algorithms and audio attributes taken from a data set of 8,000 tracks.

The aim of the project is to investigate how well different machine learning algorithms, including decision trees, support vector machines, and deep learning models, can categorize different musical genres using audio parameters including spectral properties, rhythmic patterns, and tone. In addition, clustering strategies like k-means and hierarchical clustering will be used to find connections between various genres, particularly those that overlap or have comparable traits.

The ultimate objective of this project is to develop a reliable and precise system for automatically classifying music genres that can be incorporated into music-related applications, such as playlist generators, recommendation systems, and music search engines, to enhance user experience and deliver more impartial genre classification results.

# II.	RELATED WORK
Automatic music genre classification methods now in use can be generally divided into two primary categories as Feature-based methods and Deep learning methods.

Using machine learning algorithms, feature -based methods entail collecting audio properties from music songs, such as spectral features, rhythm patterns, and tone, and categorizing genres based on these features. Decision trees, support vector machines, and k-nearest neighbors are a few of the frequently employed machine learning methods. These approaches frequently rely on manually created features and conventional machine learning techniques, which are easily understood and computationally effective. Unfortunately, they might not be able to capture subtle and complicated genre features, and they might perform worse when faced with huge and varied music data sets.

Without the requirement for manually created features, deep learning techniques use deep neural networks to learn hierarchical representations of audio features from the original raw audio data. In deep learning-based techniques for genre categorization, convolutional neural networks (CNNs) and recurrent neural networks (RNNs) are frequently employed. In capturing complex genre patterns and obtaining high accuracy in genre classification tasks, deep learning techniques have demonstrated promising results. Yet, because deep neural networks are black boxes, they may demand a lot of data for training, be expensive computationally, and be difficult to understand.

Tzanetakis and Cook’s   ”A   Survey   and   Experiments on Music Genre Classification using Audio Features and Machine Learning Algorithms” (2002), which explores various feature-based methods for music genre classification, and Oord et al”Deep content-based music recommendation” (2013), which suggests a deep learning-based method for music recommendation that incorporates genre classification, are two notable works in this field. Given the benefits and drawbacks of each approach, the development of the proposed system for automatic music genre classification in this project can be guided and inspired by these current solutions.

![2023-06-27](https://github.com/KishanGangarama/MUSIC-GENRE-CLASSIFICATION/assets/112736041/c7e31ced-aa63-40e1-a587-00df66d66cb0)

# III. OUR SOLUTION
A. Description of Data set
The Data set that will be used in this project is the free music archive dataset. It contains a sizable, carefully curated library of audio files and related metadata. About 100,000 audio tracks from diverse genres and styles—including jazz, rock, electronic, and more—make up the FMA data set. To ensure a wide variety of musical content for study, the tracks are drawn from a variety of artists and labels. The data set contains audio files in both compressed and lossless forms, making it appropriate for various analysis.

![2023-06-27 (1)](https://github.com/KishanGangarama/MUSIC-GENRE-CLASSIFICATION/assets/112736041/52a53272-58dd-4605-83fe-c494f886e362)

Along with the audio files, FMA also offers comprehensive metadata, including track details, artist information, genre tags, album art, and more. This metadata enables in-depth analysis and insights into music structure, genre classification, and other music-related tasks. It also enables academics to investigate and evaluate numerous musical qualities, such as tempo, key, mode, and instrumentation. Overall, the FMA dataset is extensive and diverse, offering a wealth of information for music analysis. This information enables researchers and analysts to investigate different facets of music, create music recommendation systems, and advance the field of music analysis through machine learning and data-driven research

B. Machine Learning Algorithms
In this study, we want to investigate several machine learning methods for categorizing music genres using audio attributes taken from the dataset. Convolutional neural networks, support vector machines, K-nearest neighbors, and clustering techniques are a few of the various algorithms we may take into account.





 


