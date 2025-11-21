from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

silhouette_scores_gmm = []
K_range = range(2, 6)

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    labels = gmm.predict(X_scaled)  # hard labels
    sil_score = silhouette_score(X_scaled, labels)
    
    silhouette_scores_gmm.append(sil_score)

plt.figure(figsize=(10,5))
plt.plot(K_range, silhouette_scores_gmm, marker='o')
plt.title("Силуэтный анализ для GMM")
plt.xlabel("Количество кластеров k")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

K_range = range(2, 6)

bic_vals = []
aic_vals = []

for k in K_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    bic_vals.append(gmm.bic(X_scaled))
    aic_vals.append(gmm.aic(X_scaled))

plt.figure(figsize=(10,5))
plt.plot(K_range, bic_vals, marker='o', label="BIC")
plt.plot(K_range, aic_vals, marker='o', label="AIC")
plt.title("AIC/BIC для GMM")
plt.xlabel("Количество кластеров k")
plt.ylabel("Значение критерия")
plt.legend()
plt.grid(True)
plt.show()


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)
labels_gmm = gmm.predict(X_scaled)

print(adjusted_rand_score(y_true, labels_gmm))
print(jaccard_score(y_true, labels_gmm, average='macro'))

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca[:,0], y=df_pca[:,1], hue=labels_gmm, palette='Set1')
plt.title('GMM')
plt.show()

sil = silhouette_score(df_scaled, labels_gmm)
print(sil)