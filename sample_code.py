import pandas
import networkx as nx
import scipy
import clean_original_dataset_HK
import clean_merge_HK_template

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

############################ FUNCTIONS
def scaling_data(df, features):
	scaler = StandardScaler()
	X = scaler.fit_transform(df[features])
	return X


def weighted_projected_graph(B, nodes, ratio=False):
    if B.is_directed():
        pred = B.pred
        G = nx.DiGraph()
    else:
        pred = B.adj
        G = nx.Graph()
    G.graph.update(B.graph)
    G.add_nodes_from((n, B.nodes[n]) for n in nodes)
    n_top = float(len(B) - len(nodes))
    nodes_checked = []
    for u in nodes:
        nodes_checked.append(u)
        unbrs = set(B[u])
        nbrs2 = {n for nbr in unbrs for n in B[nbr]} - set(nodes_checked)
        for v in nbrs2:
            vnbrs = set(pred[v])
            common = unbrs & vnbrs
            if not ratio:
                weight = len(common)
            else:
                weight = len(common) / n_top
            G.add_edge(u, v, weight=weight)
    return G


def obtain_network_features(reviews):

	# initializing the product-level data
	df = pandas.DataFrame({"asin": reviews.asin.unique()})

	# building the bipartite product-reviewer graph
	B = nx.Graph()
	B.add_nodes_from(reviews.reviewerID, bipartite=0)
	B.add_nodes_from(reviews.asin, bipartite=1)
	B.add_edges_from([(row['reviewerID'], row['asin'])
	                 for idx, row in reviews.iterrows()])

	# building the product projected graph
	P = weighted_projected_graph(B, reviews.asin.unique())
	print("finish graph creating")

	w_degree_cent = nx.degree(P, weight='weight')
	eig_cent = nx.eigenvector_centrality(P, max_iter=500)
	pr = nx.pagerank(P, alpha=0.85)
	cc = nx.clustering(P)

	# creating the features data
	df['pagerank_org'] = [pr[i] for i in df.asin]
	df['eigenvector_cent_org'] = [eig_cent[i] for i in df.asin]
	df['clustering_coef_org'] = [cc[i] for i in df.asin]
	df['w_degree_org'] = [w_degree_cent[i] for i in df.asin]

	return df

def classification_results(df_train, df_test, features):

	X_train = df_train[features].values
	y_train = df_train['fake'].values
	X_test = df_test[features].values

	print("Shape of train and test:",X_train.shape, X_test.shape)

	model = RandomForestClassifier(random_state=42, 
	                               n_estimators=1200,
	                               min_samples_leaf=3,
	                               min_samples_split=6,
	                               max_features='auto',
	                               max_depth=40,
	                               bootstrap=True,
	                               n_jobs=-1)
	model.fit(X_train, y_train)
	y_prob_pred = model.predict_proba(X_test)[:,1]
	print(sum(y_prob_pred >= 0), sum(y_prob_pred >= 0.5), sum(y_prob_pred >= 0.6), sum(y_prob_pred >= 0.7))

	df_test['p_fake'] = y_prob_pred
	return df_test



################################# CLEAN DATA
# reviews
reviews = pandas.read_csv("parsed_files/HK_reviews.csv", lineterminator='\n') 
reviews = clean_original_dataset_HK.clean_original_dataset(reviews)
print("Number of products, reviews, and reviewers in reviews dataset:", \
				len(reviews.asin.unique()),\
				reviews.shape[0],\
				len(reviews.reviewerID.unique()))

# Amazon product level data
merge_HK = pandas.read_csv('parsed_files/HK_variables_on_product_level.csv', lineterminator='\n')
merge_HK = clean_merge_HK_template.clean_merge_HK(merge_HK)

# Sherry He's data
df_ours = pandas.read_csv('parsed_files/product_level_data_without_img_feats.csv.gz')


################################## CLUSTERING
review_features = [
		'avg_sim_TF_IDF',
        'overall_asin_mean', 
        'one_star_share','five_star_share','vote_share','words_count_new_std',
        'image_asin_mean', 
        'average_gap_days', 'min_gap_days','max_gap_days','stdev_gap_days'
		]
network_features = ['pagerank_org','eigenvector_cent_org', 'w_degree_org', 'clustering_coef_org']

features_to_use = review_features + network_features

X = scaling_data(merge_HK, features_to_use)
k = 20
method = KMeans(n_clusters=k, random_state=42).fit(X)
labels = method.labels_
merge_HK['cluster_ID'] = labels + 1


################################# CLASSIFICATION ON CLUSTERS
frames = []
for i in range(k):

	print("================ CLUSTER {}====================".format(i+1))
	# obtain the network features
	df_network = obtain_network_features(reviews.loc[reviews.asin.isin(merge_HK.loc[merge_HK.cluster_ID == i+1,'asin'].values), :])

	# obtain all features
	df = df_network[['asin'] + network_features].merge(merge_HK[review_features+['asin']], on='asin', how='inner')

	# classify
	df_with_p_fake = classification_results(df_ours, df, features=features_to_use)

	# append the data
	frames.append(df_with_p_fake)

# combining all clusters in one df
clusters = pandas.concat(frames, axis=0, ignore_index=True)
clusters = clusters.merge(merge_HK[['asin', 'cluster_ID']], on='asin', how='inner')

################################ RESULTS
clusters_pt = clusters.pivot_table(index='cluster_ID', aggfunc={'clustering_coef_org': 'mean',
																'eigenvector_cent_org': 'mean',
																'image_asin_mean': 'mean',
																'w_degree_org': 'mean',
																'max_gap_days':'mean',
																'pagerank_org':'mean',
																'five_star_share':'mean',
																'avg_sim_TF_IDF':'mean', 
																'average_gap_days':'mean', 'stdev_gap_days':'mean', 'overall_asin_mean':'mean', 
																'words_count_new_std':'mean', 'one_star_share':'mean', 'vote_share':'mean',
																'min_gap_days':'mean', 'asin':'count', 'p_fake':lambda x:(x>=0.5).sum(),})
clusters_pt[review_features + network_features] = scipy.stats.zscore(clusters_pt[review_features + network_features])
clusters_pt = clusters_pt.reindex(['clustering_coef_org','eigenvector_cent_org',
									'image_asin_mean','w_degree_org',
									'max_gap_days',
									'pagerank_org','five_star_share',
									'avg_sim_TF_IDF',
									'average_gap_days',
									'stdev_gap_days','overall_asin_mean','words_count_new_std',
									'one_star_share','vote_share','min_gap_days','asin','p_fake'], axis=1)
clusters_pt


merge_HK_all = clusters[['asin', 'p_fake']].merge(merge_HK, on='asin', how='inner')
merge_HK_all['p_fake'] = round(merge_HK_all['p_fake'], 5)
merge_HK_all.to_stata('parsed_files/final_dataset.dta', version=117)