# DiscourseOverrule
Model of temporal-spatial integration of wikipedia2vec embeddings using easyesn to simulate human discourse processing
This corresponds to the article: 

Uchida, T., Lair, N., Ishiguro, H., & Dominey, P. F. (2021). A Model of Online Temporal-Spatial Integration for Immediacy and Overrule in Discourse Comprehension. Neurobiology of Language, 2(1), 83-105.

We employ vector cosine as a method to calculate semantic relatedness, and in an update we use this to predict N400 responses as N400 = 1 - semantic relatedness.  There is a branch of the repository that implements the N400 transformation.
