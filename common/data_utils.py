
data_dir = "/Users/shenfan/Code/Project/GraphWorld/data/"

edge_file = {"facebook": "facebook/facebook.txt",
             "arxiv": "arxiv/arxiv.txt",
             "BlogCatalog": "BlogCatalog/edges_format2.txt",
             "lyb": "lyb/lyb.e",
             "USAir": "USAir/USAir.txt",
             "twitter": "twitter/twitter.txt",
             "MUTAG": "MUTAG/MUTAG.txt",
             "IMDBBINARY": "IMDBBINARY/IMDBBINARY.txt",
             "PTC": "PTC/PTC.txt"
             }

node_file = {"lyb" : "lyb/lyb.v"}

def get_data_path(dataset_name):
    return data_dir + edge_file[dataset_name]

def get_node_path(dataset_name):
    return data_dir + node_file[dataset_name]

if __name__ == "__main__":
    print(get_data_path("facebook"))
