import streamlit as st
import torch
from feature_statistics import FeatureStatistics
import html
from auto_interp import AutoInterpreter
import plotly.graph_objects as go
import numpy as np
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer
import requests


# Load the tokenizer and dataset
@st.cache_resource
def load_tokenizer_and_dataset(dataset="NeelNanda/pile-10k", context_size=128, prepend_bos=True):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = torch.load("dataset.pth")
    return tokenizer, dataset

# Load the FeatureStatistics object
@st.cache_resource
def load_feature_statistics(filename):
    return FeatureStatistics.load(filename, None)

# Load the AutoInterpreter
@st.cache_resource
def load_interpreter(_stats, _tokenizer, _dataset):
    interpreter = AutoInterpreter(_stats, _tokenizer, _dataset)
    try:
        interpreter.load_descriptions("feature_descriptions.json")
    except FileNotFoundError:
        pass
    return interpreter

def meta_feature_button_callback(meta_feature_idx):
    st._set_query_params(page="Meta Feature Explorer", meta_feature=meta_feature_idx)
    st.session_state.page = "Meta Feature Explorer"
    st.session_state.meta_feature_idx = meta_feature_idx

def feature_button_callback(feature_idx):
    st._set_query_params(page="Feature Explorer", feature=feature_idx)
    st.session_state.page = "Feature Explorer"
    st.session_state.feature_idx = feature_idx


def create_radial_tree_plot(feature_idx, stats, interpreter, model_id="gpt2-small", layer="8-res_fs49152-jb"):
    # Create nodes and edges
    nodes = [f"F{feature_idx}"]
    node_colors = ['rgba(255,0,0,0.3)']  # More transparent red
    node_sizes = [25]
    edge_traces = []
    annotations = []

    feature_desc = interpreter.get_neuronpedia_explanation(feature_idx, model_id, layer)
    annotations.append(dict(x=0, y=0, xref="x", yref="y", text=feature_desc, showarrow=False, font=dict(size=12, color="red")))

    sorted_meta_features = sorted(stats.feature_to_clusters[feature_idx], key=lambda x: x[1], reverse=True)[:5]
    for i, (mf_idx, _) in enumerate(sorted_meta_features):
        mf_desc = interpreter.interpret_meta_feature(mf_idx)  # Use AutoInterpreter for meta-features
        mf_node = f"MF{mf_idx}"
        nodes.append(mf_node)
        node_colors.append('rgba(0,0,255,0.3)')  # More transparent blue
        node_sizes.append(20)

        angle = 2 * np.pi * i / 5
        x, y = 0.5 * np.cos(angle), 0.5 * np.sin(angle)
        edge_traces.append(go.Scatter(x=[0, x], y=[0, y], mode='lines', line=dict(color='rgba(136, 136, 136, 0.5)', width=1), hoverinfo='none'))
        annotations.append(dict(x=x, y=y, xref="x", yref="y", text=mf_desc, showarrow=False, font=dict(size=10, color="blue")))

        top_features = sorted(stats.cluster_to_features[mf_idx], key=lambda x: x[1], reverse=True)[:5]
        for j, (f_idx, _) in enumerate(top_features):
            if f_idx != feature_idx:
                f_desc = interpreter.get_neuronpedia_explanation(f_idx, model_id, layer)
                f_node = f"F{f_idx}_{i}"
                nodes.append(f_node)
                node_colors.append('rgba(0,255,0,0.3)')  # More transparent green
                node_sizes.append(15)

                sub_angle = angle + (j - 2) * 0.2
                sub_x, sub_y = (0.8 + 0.05 * j) * np.cos(sub_angle), (0.8 + 0.05 * j) * np.sin(sub_angle)
                edge_traces.append(go.Scatter(x=[x, sub_x], y=[y, sub_y], mode='lines', line=dict(color='rgba(136, 136, 136, 0.3)', width=1), hoverinfo='none'))
                annotations.append(dict(x=sub_x, y=sub_y, xref="x", yref="y", text=f_desc, showarrow=False, font=dict(size=8, color="green")))

    # Create node trace
    node_trace = go.Scatter(
        x=[ann['x'] for ann in annotations],
        y=[ann['y'] for ann in annotations],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='rgba(255,255,255,0.5)')
        ),
        text=[ann['text'] for ann in annotations],
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=f'Feature {feature_idx} Relationship Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=annotations,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),
                        width=800,
                        height=800,
                        plot_bgcolor='rgba(0,0,0,0)'
                    ))

    return fig


def format_example(example, tokenizer):
    context = html.escape(example['context_text'])
    token = html.escape(tokenizer.decode(example['token']))
    
    token_start = context.lower().rfind(token.lower())
    if token_start != -1:
        token_end = token_start + len(token)
        formatted_context = f"{context[:token_start]}<b>{context[token_start:token_end]}</b>{context[token_end:]}"
    else:
        formatted_context = f"{context} (Token: <b>{token}</b>)"
    
    return formatted_context

def main():
    st.title("Feature Explorer Dashboard")

    # Get query parameters from URL
    query_params = st._get_query_params()
    page = query_params.get("page", ["Feature Explorer"])[0]
    feature_idx = int(query_params.get("feature", [0])[0])
    meta_feature_idx = int(query_params.get("meta_feature", [0])[0])

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = page
    if 'meta_feature_idx' not in st.session_state:
        st.session_state.meta_feature_idx = meta_feature_idx
    if 'feature_idx' not in st.session_state:
        st.session_state.feature_idx = feature_idx

    # Load everything
    tokenizer, dataset = load_tokenizer_and_dataset()
    stats = load_feature_statistics("feature_stats_gpt2.pth")
    interpreter = load_interpreter(stats, tokenizer, dataset)

    # Sidebar for navigation
    st.sidebar.radio("Choose a page", ["Feature Explorer", "Meta Feature Explorer"], key="page", index=["Feature Explorer", "Meta Feature Explorer"].index(st.session_state.page))

    if st.session_state.page == "Feature Explorer":
        st.header("Feature Explorer")
        
        feature_idx = st.number_input("Enter feature index:", min_value=0, max_value=stats.n_features-1, value=st.session_state.feature_idx)
        
        if st.button("Explore Feature") or feature_idx != st.session_state.feature_idx:
            st.session_state.feature_idx = feature_idx
            st._set_query_params(page="Feature Explorer", feature=feature_idx)
            st.rerun()

        # Embed Neuronpedia iframe
        st.components.v1.iframe(
            f"https://neuronpedia.org/gpt2-small/8-res_fs49152-jb/{st.session_state.feature_idx}?embed=true&embedexplanation=true&embedplots=true",
            height=1200,
            scrolling=False
        )

        # Display associated meta features (sorted by activation)
        st.write("## Associated Meta Features:")
        sorted_meta_features = sorted(stats.feature_to_clusters[st.session_state.feature_idx], key=lambda x: x[1], reverse=True)
        for meta_feature_idx, activation in sorted_meta_features:
            meta_feature_description = interpreter.interpret_meta_feature(meta_feature_idx)
            st.button(f"Meta Feature {meta_feature_idx}: {meta_feature_description} (Activation: {activation:.4f})", 
                      key=f"meta_feature_{meta_feature_idx}", 
                      on_click=meta_feature_button_callback, 
                      args=(meta_feature_idx,))
            
        st.write("## Feature Relationship Graph")
        with st.spinner("Generating graph..."):
            fig = create_radial_tree_plot(st.session_state.feature_idx, stats, interpreter)
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.page == "Meta Feature Explorer":
        st.header("Meta Feature Explorer")
        
        meta_feature_idx = st.number_input("Enter meta feature index:", min_value=0, value=st.session_state.meta_feature_idx)
        
        if st.button("Explore Meta Feature") or meta_feature_idx != st.session_state.meta_feature_idx:
            st.session_state.meta_feature_idx = meta_feature_idx
            st._set_query_params(page="Meta Feature Explorer", meta_feature=meta_feature_idx)
            st.rerun()
            
        # Compute interpretation for the current meta-feature
        with st.spinner("Generating description..."):
            meta_feature_description = interpreter.interpret_meta_feature(st.session_state.meta_feature_idx)
        st.write(f"# Meta Feature {st.session_state.meta_feature_idx}: {meta_feature_description}")
        
        if st.session_state.meta_feature_idx not in stats.cluster_to_features:
            st.write(f"Meta Feature {st.session_state.meta_feature_idx} not found.")
        else:
            features = stats.cluster_to_features[st.session_state.meta_feature_idx]
            features.sort(key=lambda x: x[1], reverse=True)
            
            st.write(f"Number of features in this meta feature: {len(features)}")
            
            top_k = 5
            
            for feature_idx, activation in features[:top_k]:
                # feature_description = interpreter.interpret_feature(feature_idx)
                # st.write(f"## Feature {feature_idx}: {feature_description} (activation: {activation:.4f}):")
                st.button(f"Explore Feature {feature_idx}", 
                          key=f"feature_{feature_idx}", 
                          on_click=feature_button_callback, 
                          args=(feature_idx,))
                
                # Embed Neuronpedia iframe for each feature
                st.components.v1.iframe(
                    f"https://neuronpedia.org/gpt2-small/8-res_fs49152-jb/{feature_idx}?embed=true&embedexplanation=true&embedplots=true",
                    height=900,
                    scrolling=False
                )
                
                st.markdown("", unsafe_allow_html=True)

if __name__ == "__main__":
    main()