import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

dir_path = "gpt5-mini"
RESULTS_PATH = f'output_data/{dir_path}/results_seedprompt0.json'
PROMPT_ANALYSIS_PATH = f'output_data/{dir_path}/analysis/'
os.makedirs(PROMPT_ANALYSIS_PATH, exist_ok=True)

MAX_TITLE_WIDTH = 40
helpful_score = 2


def wrap_title(title, width=40):
    return "\n".join(textwrap.wrap(title, width))


def fmt_score(score):
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "None"


with open(RESULTS_PATH) as f:
    results = json.load(f)


# ---------- weighted layout helpers ----------

def compute_leaf_counts(G, root):
    """
    Compute number of leaf descendants for each node in a tree.
    Leaves have count 1.
    """
    leaf_counts = {}

    def dfs(node):
        children = list(G.successors(node))
        if not children:
            count = 1
        else:
            count = sum(dfs(c) for c in children)
        leaf_counts[node] = count
        return count

    dfs(root)
    return leaf_counts


def hierarchy_pos_weighted(G, root, leaf_counts,
                           width=1.8, vert_gap=0.7, vert_loc=0,
                           xcenter=0.5, pos=None):
    """
    Hierarchical layout where each child's horizontal span is
    proportional to the size of its subtree (leaf_counts).
    This helps when some nodes have many children but most
    nodes at that level stop.
    """
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)

    children = list(G.successors(root))
    if not children:
        return pos

    total_leafs = sum(leaf_counts[c] for c in children)
    start_x = xcenter - width / 2.0
    curr_x = start_x

    for child in children:
        child_width = width * (leaf_counts[child] / total_leafs)
        child_center_x = curr_x + child_width / 2.0

        pos = hierarchy_pos_weighted(
            G,
            child,
            leaf_counts,
            width=child_width,
            vert_gap=vert_gap,
            vert_loc=vert_loc - vert_gap,
            xcenter=child_center_x,
            pos=pos
        )
        curr_x += child_width

    return pos


def create_analysis_tree(data, title, output_path):
    # Build the directed graph
    G = nx.DiGraph()

    for node in data:
        node_id = node["id"]
        parent_id = node["parent_id"]
        score = node["score"]

        # Add node with score as an attribute
        G.add_node(node_id, score=score)

        # Add edge from parent to child (skip if root)
        if parent_id is not None:
            G.add_edge(parent_id, node_id)

    # Find the root (node with parent_id == None)
    roots = [n["id"] for n in data if n["parent_id"] is None]
    if not roots:
        raise ValueError("No root found (no node with parent_id == None)")
    root = roots[0]

    # Compute leaf counts and positions for all nodes (weighted layout)
    leaf_counts = compute_leaf_counts(G, root)
    pos = hierarchy_pos_weighted(G, root, leaf_counts)

    # Build labels: show id and score
    labels = {
        n: f"ID={n}\nS={fmt_score(G.nodes[n]['score'])}"
        for n in G.nodes()
    }

    # ---- edge colors based on parent/child score ----
    edge_colors = []
    for parent, child in G.edges():
        parent_score = G.nodes[parent].get("score")
        child_score = G.nodes[child].get("score")

        color = "black"
        try:
            p = float(parent_score)
            c = float(child_score)
            if c < p:
                color = "red"
        except Exception:
            pass

        edge_colors.append(color)
    # -------------------------------------------------

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=False,
        arrows=True,
        node_size=1200,
        edge_color=edge_colors,  # use the colors we computed
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title(wrap_title(title, MAX_TITLE_WIDTH), fontsize=14, pad=20)
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_filtered_query_summary_tree(prompts, title, output_path):
    """
    Create a tree over prompts (using prompt parent_id) where each node label is:
        prompt_id and n_filtered_root_queries / n_total_root_queries.
    Only queries with parent_id == None are counted.
    """
    G = nx.DiGraph()

    # Build nodes with filtered / total counts
    for prompt in prompts:
        pid = prompt["id"]
        parent_pid = prompt["parent_id"]

        total_root_queries = 0
        filtered_root_queries = 0

        # prompt["queries"] is a list of query-lists
        for query_list in prompt.get("queries", []):
            for q in query_list:
                if q.get("parent_id") is None:
                    total_root_queries += 1
                    if q.get("is_filtered_out"):
                        filtered_root_queries += 1

        # store counts as attributes
        G.add_node(pid,
                   filtered=filtered_root_queries,
                   total=total_root_queries)

        # connect prompts according to their parent_id
        if parent_pid is not None:
            G.add_edge(parent_pid, pid)

    # Find root prompt (parent_id == None)
    roots = [p["id"] for p in prompts if p["parent_id"] is None]
    if not roots:
        raise ValueError("No root prompt found (no prompt with parent_id == None)")
    root = roots[0]

    # Compute leaf counts & layout
    leaf_counts = compute_leaf_counts(G, root)
    pos = hierarchy_pos_weighted(G, root, leaf_counts)

    # Labels: prompt id and filtered/total
    labels = {
        n: f"Prompt={n}\nF={G.nodes[n]['filtered']}/{G.nodes[n]['total']}"
        for n in G.nodes()
    }

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=1200)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    plt.title(wrap_title(title, MAX_TITLE_WIDTH), fontsize=14, pad=20)
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_unpreferred_query_summary_tree(prompts, title, output_path):
    
    G = nx.DiGraph()

    # Build nodes with filtered / total counts
    for prompt in prompts:
        pid = prompt["id"]
        parent_pid = prompt["parent_id"]

        total_queries = 0
        unpreferred_queries = 0

        # prompt["queries"] is a list of query-lists
        for query_list in prompt.get("queries", []):
            for q in query_list:
                if q.get("is_filtered_out") == False and q.get("score") != "None":
                    total_queries += 1
                    if float(q.get("score")) < helpful_score:
                        unpreferred_queries += 1

        # store counts as attributes
        G.add_node(pid,
                   unpreferred=unpreferred_queries,
                   total=total_queries)

        # connect prompts according to their parent_id
        if parent_pid is not None:
            G.add_edge(parent_pid, pid)

    # Find root prompt (parent_id == None)
    roots = [p["id"] for p in prompts if p["parent_id"] is None]
    if not roots:
        raise ValueError("No root prompt found (no prompt with parent_id == None)")
    root = roots[0]

    # Compute leaf counts & layout
    leaf_counts = compute_leaf_counts(G, root)
    pos = hierarchy_pos_weighted(G, root, leaf_counts)

    # Labels: prompt id and unpreferred/total
    labels = {
        n: f"Prompt={n}\nN={G.nodes[n]['unpreferred']}/{G.nodes[n]['total']}"
        for n in G.nodes()
    }

    # Draw
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=4800)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    plt.title(wrap_title(title, MAX_TITLE_WIDTH), fontsize=14, pad=20)
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# prompt analysis
create_analysis_tree(results, "Prompt Analysis", PROMPT_ANALYSIS_PATH + "prompt_analysis.png")
create_unpreferred_query_summary_tree(
    results,
    "Prompt Unpreferred Summary",
    PROMPT_ANALYSIS_PATH + "prompt_unpreferred_summary.png"
)
create_filtered_query_summary_tree(
    results,
    "Prompt Filtered Query Summary",
    PROMPT_ANALYSIS_PATH + "prompt_filtered_summary.png"
)
query_n = 0
prompt_n = 0
for prompt in results:
    for queries in prompt["queries"]:
        roots = [q for q in queries if q["parent_id"] is None]
        query_root = roots[0]
        create_analysis_tree(
            queries,
            query_root["query"],
            PROMPT_ANALYSIS_PATH + f"prompt_analysis_prompt{prompt_n}_query{query_n}.png"
        )
        query_n += 1
    prompt_n += 1
