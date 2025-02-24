from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)

            # Preprocess data
            numerical_cols = ['age', 'tenure', 'monthly_spending', 'num_products']
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

            # Perform Clustering
            X = df[numerical_cols]
            clustering = AgglomerativeClustering(n_clusters=4)
            df['cluster'] = clustering.fit_predict(X)

            # Save clustered data
            clustered_filepath = os.path.join(UPLOAD_FOLDER, 'clustered_data.csv')
            df.to_csv(clustered_filepath, index=False)

            # Generate visualization file paths
            dendro_path = os.path.join(UPLOAD_FOLDER, 'dendrogram.png')
            scatter_path = os.path.join(UPLOAD_FOLDER, 'scatter.png')
            hist_path = os.path.join(UPLOAD_FOLDER, 'histograms.png')
            pairplot_path = os.path.join(UPLOAD_FOLDER, 'pairplot.png')

            # 🔥 Generate Dendrogram
            plt.figure(figsize=(10, 7))
            linked = linkage(X, 'ward')
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
            plt.title('Dendrogram')
            plt.xlabel('Customer Index')
            plt.ylabel('Distance')
            plt.savefig(dendro_path)
            plt.close()

            # 🔥 Generate Scatter Plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x='monthly_spending', y='age', hue='cluster', palette='viridis')
            plt.title('Customer Clusters')
            plt.savefig(scatter_path)
            plt.close()

            # 🔥 Generate Histograms
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot(2, 2, i)
                sns.histplot(df[col], bins=30, kde=True)
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            # 🔥 Generate Pairplot
            pairplot = sns.pairplot(df, hue='cluster', vars=numerical_cols)
            pairplot.savefig(pairplot_path)
            plt.close()

            # Cluster analysis tables
            cluster_analysis = df.groupby('cluster')[numerical_cols].mean().to_html(classes='table table-striped', border=0)
            summary_stats = df.groupby('cluster')[numerical_cols].agg(['mean', 'median', 'std']).to_html(classes='table table-striped', border=0)

            return render_template('result.html', 
                                   clustered_data=clustered_filepath, 
                                   dendrogram_img=dendro_path, 
                                   scatter_img=scatter_path, 
                                   hist_img=hist_path,
                                   pairplot_img=pairplot_path,
                                   cluster_analysis=cluster_analysis,
                                   summary_stats=summary_stats)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
