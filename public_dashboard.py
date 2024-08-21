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
import random
import plotly.figure_factory as ff


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
    feature_link = f"https://metasae.streamlit.app/?page=Feature+Explorer&feature={feature_idx}"
    annotations.append(dict(x=0, y=0, xref="x", yref="y", text=f'<a href="{feature_link}" style="color: red;">{feature_desc}</a>', showarrow=False, font=dict(size=14, color="red")))

    sorted_meta_features = sorted(stats.feature_to_clusters[feature_idx], key=lambda x: x[1], reverse=True)[:5]
    top_features_dict = {}

    for i, (mf_idx, _) in enumerate(sorted_meta_features):
        mf_desc = interpreter.interpret_meta_feature(mf_idx)  # Use AutoInterpreter for meta-features
        mf_node = f"MF{mf_idx}"
        nodes.append(mf_node)
        node_colors.append('rgba(0,0,255,0.3)')  # More transparent blue
        node_sizes.append(20)

        angle = 2 * np.pi * i / 5 + 1
        x, y =   0.5*np.cos(angle),  0.5*np.sin(angle)
        edge_traces.append(go.Scatter(x=[0, x], y=[0, y], mode='lines', line=dict(color='rgba(136, 136, 136, 0.5)', width=1), hoverinfo='none'))
        mf_link = f"https://metasae.streamlit.app/?page=Meta+Feature+Explorer&meta_feature={mf_idx}"
        annotations.append(dict(x=x, y=y, xref="x", yref="y", text=f'<a href="{mf_link}" style="color: blue;">{mf_desc}</a>', showarrow=False, font=dict(size=12, color="blue")))

        top_features = sorted(stats.cluster_to_features[mf_idx], key=lambda x: x[1], reverse=True)[:5]
        top_features_dict[mf_idx] = top_features
        for j, (f_idx, _) in enumerate(top_features):
            if f_idx != feature_idx:
                f_desc = interpreter.get_neuronpedia_explanation(f_idx, model_id, layer)
                f_node = f"F{f_idx}_{i}"
                nodes.append(f_node)
                node_colors.append('rgba(0,255,0,0.3)')  # More transparent green
                node_sizes.append(15)

                sub_angle = angle + (j - 2) * 0.2
                sub_x, sub_y = 1.1*(0.8 + 0.05 * j) * np.cos(sub_angle), 1.1*(0.8 + 0.05 * j) * np.sin(sub_angle)
                edge_traces.append(go.Scatter(x=[x, sub_x], y=[y, sub_y], mode='lines', line=dict(color='rgba(136, 136, 136, 0.3)', width=1), hoverinfo='none'))
                f_link = f"https://metasae.streamlit.app/?page=Feature+Explorer&feature={f_idx}"
                annotations.append(dict(x=sub_x, y=sub_y, xref="x", yref="y", text=f'<a href="{f_link}" style="color: green;">{f_desc}</a>', showarrow=False, font=dict(size=10, color="green")))

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

    return fig, sorted_meta_features, top_features_dict


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

def create_activation_histogram(activations, num_features):
    fig = ff.create_distplot([activations], ['Activation'], bin_size=0.05, show_rug=False, show_curve=False)
    fig.update_layout(
        title=f"Meta-Feature Activation Distribution<br>Number of features: {num_features} (activation density: {(num_features/49152):.4f})",
        xaxis_title="Activation Value",
        yaxis_title="Density",
        showlegend=False,
        width=400,
        height=350,
        xaxis_range=[0.15, 1]
    )
    return fig

def main():
    st.title("Meta SAE Dashboard")

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

    # Add MetaSAE explanation to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About MetaSAEs")
    st.sidebar.markdown("""
    MetaSAEs are sparse autoencoders (SAEs) trained on the decoder directions of another SAE. 
    They decompose SAE features into more interpretable components (meta-features), revealing deeper structures in the activation space.
        
    This MetaSAE is trained on a 49152-feature SAE and has a dictionary size of 2304 meta-features.
                        
    It is trained with a BatchTopK of 4, such that on average every feature decomposes into 4 meta-features.
    """)

    if st.session_state.page == "Feature Explorer":
        st.header("Feature Explorer")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            feature_idx = st.number_input("Enter feature index:", min_value=0, max_value=stats.n_features-1, value=st.session_state.feature_idx)
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Explore Feature", key="explore_feature_button"):
                st.session_state.feature_idx = feature_idx
                st._set_query_params(page="Feature Explorer", feature=feature_idx)
                st.rerun()
        
        with col2:
            if st.button("I'm Feeling Lucky", key="feature_lucky_button"):
                feature_idx = random.randint(0, stats.n_features-1)
                st.session_state.feature_idx = feature_idx
                st._set_query_params(page="Feature Explorer", feature=feature_idx)
                st.rerun()
        

        # Embed Neuronpedia iframe
        st.components.v1.iframe(
            f"https://neuronpedia.org/gpt2-small/8-res_fs49152-jb/{st.session_state.feature_idx}?embed=true&embedexplanation=true&embedplots=true",
            height=600,
            scrolling=True
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
            
        st.write("## Related Features Graph")
        with st.spinner("Generating graph..."):
            fig, sorted_meta_features, top_features_dict = create_radial_tree_plot(st.session_state.feature_idx, stats, interpreter)
            st.plotly_chart(fig, use_container_width=True)


        num_columns = len(sorted_meta_features)
        columns = st.columns(num_columns)

        # for i, (mf_idx, _) in enumerate(sorted_meta_features):
        #     with columns[i]:
        #         st.write(f"**Meta Feature {mf_idx}**")
        #         for f_idx, _ in top_features_dict[mf_idx]:
        #             if f_idx != st.session_state.feature_idx:
        #                 f_desc = interpreter.get_neuronpedia_explanation(f_idx, "gpt2-small", "8-res_fs49152-jb")
        #                 if st.button(f"F{f_idx}: {f_desc}", key=f"related_feature_{f_idx}"):
        #                     st.session_state.feature_idx = f_idx
        #                     st._set_query_params(page="Feature Explorer", feature=f_idx)
        #                     st.rerun()

    elif st.session_state.page == "Meta Feature Explorer":
        st.header("Meta Feature Explorer")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            meta_feature_idx = st.number_input("Enter meta feature index:", min_value=0, value=st.session_state.meta_feature_idx)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Explore Meta Feature", key="explore_meta_feature_button"):
                st.session_state.meta_feature_idx = meta_feature_idx
                st._set_query_params(page="Meta Feature Explorer", meta_feature=meta_feature_idx)
                st.rerun()
        
        with col2:
            if st.button("I'm Feeling Lucky", key="meta_feature_lucky_button"):
                meta_feature_idx = random.choice(list(stats.cluster_to_features.keys()))
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
            
            # st.write(f"Number of features with this meta feature: {len(features)} (activation density: {(len(features)/49152):.4f})")


            # Create and display the activation histogram
            activations = [activation for _, activation in features]
            fig = create_activation_histogram(activations, len(features))
            st.plotly_chart(fig, use_container_width=True)

            # Display top boosted logits for the meta-feature
            st.write("#### Top Boosted Logits for Meta Feature:")
            meta_top_logits = stats.meta_feature_top_boosted_logits.get(st.session_state.meta_feature_idx, [])
            st.write(", ".join([f"{token}" for token, value in meta_top_logits[:10]]))

            st.write("## Max activating features")
            # Initialize session state for number of features to show
            if 'num_features_to_show' not in st.session_state:
                st.session_state.num_features_to_show = 5

            for feature_idx, activation in features[:st.session_state.num_features_to_show]:
                st.button(f"Explore Feature {feature_idx}", 
                          key=f"feature_{feature_idx}", 
                          on_click=feature_button_callback, 
                          args=(feature_idx,))
                
                # Embed Neuronpedia iframe for each feature
                st.components.v1.iframe(
                    f"https://neuronpedia.org/gpt2-small/8-res_fs49152-jb/{feature_idx}?embed=true&embedexplanation=true&embedplots=false",
                    height=400,
                    scrolling=True
                )
                
                st.markdown("", unsafe_allow_html=True)

            # Add "Load More" button
            if st.session_state.num_features_to_show < len(features) and st.button("Load More"):
                st.session_state.num_features_to_show += 10
                st.rerun()

if __name__ == "__main__":
    main()