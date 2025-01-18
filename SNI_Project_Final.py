import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# Step 1: Load the Twitter dataset
file_path = '/Users/devikawandhare/Documents/SNI/twitter_new.csv'

# Load the dataset as a Pandas DataFrame
try:
    edges_df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Network Construction
print("Constructing the network...")
# Assume the dataset has columns 'source' and 'target' for directed edges
G = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.DiGraph())

# Step 3: Community Detection
print("Detecting communities...")
# Using the Girvan-Newman method for demonstration (can be replaced with Louvain)
# Convert to undirected graph for community detection
G_undirected = G.to_undirected()
communities = list(greedy_modularity_communities(G_undirected))
print(f"Number of communities detected: {len(communities)}")

# Create a dictionary to map each node to its community
community_map = {}
for i, community in enumerate(communities):
    for node in community:
        community_map[node] = i

# Step 4: Centrality Analysis
print("Calculating centrality measures...")
# Calculate Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
# Calculate Eigenvector Centrality (on undirected graph)
eigenvector_centrality = nx.eigenvector_centrality(G_undirected, max_iter=1000)

# Find top influencers based on centrality
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top Influential Users by Betweenness Centrality:")
for user, score in top_betweenness:
    print(f"User: {user}, Score: {score:.4f}")

print("\nTop Influential Users by Eigenvector Centrality:")
for user, score in top_eigenvector:
    print(f"User: {user}, Score: {score:.4f}")

# Step 5: Visualization and Interpretation
print("Visualizing the network...")
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)

# Assign colors to communities
colors = [community_map[node] if node in community_map else 0 for node in G.nodes()]

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Twitter Network with Community Detection")
plt.show()
