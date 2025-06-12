import streamlit as st
import numpy as np
import functools
import operator
import pygad
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="GA Image Reproducer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

def img2chromosome(img_arr):
    return np.reshape(img_arr, (functools.reduce(operator.mul, img_arr.shape)))

def chromosome2img(vector, shape):
    if len(vector) != functools.reduce(operator.mul, shape):
        raise ValueError(f"Vector length {len(vector)} doesn't match shape {shape}")
    return np.reshape(vector, shape)

def fitness_function(ga_instance, solution, solution_idx):
    target_chromosome = st.session_state.target_chromosome
    fitness = np.sum(np.abs(target_chromosome - solution))
    fitness = np.sum(target_chromosome) - fitness
    return fitness

def on_generation_callback(ga_instance):
    if 'fitness_history' not in st.session_state:
        st.session_state.fitness_history = []

    current_fitness = ga_instance.best_solution()[1]
    st.session_state.fitness_history.append(current_fitness)

    progress = ga_instance.generations_completed / st.session_state.max_generations
    st.session_state.progress_bar.progress(progress)

    if ga_instance.generations_completed % 10 == 0:
        col1, col2, col3 = st.session_state.metrics_cols
        with col1:
            st.metric("Generation", ga_instance.generations_completed)
        with col2:
            st.metric("Best Fitness", f"{current_fitness:.2f}")
        with col3:
            improvement = 0
            if len(st.session_state.fitness_history) > 1:
                improvement = current_fitness - st.session_state.fitness_history[-2]
            st.metric("Improvement", f"{improvement:.4f}")

    if ga_instance.generations_completed % 100 == 0 and ga_instance.generations_completed > 0:
        best_solution = ga_instance.best_solution()[0]
        result_img = chromosome2img(best_solution, st.session_state.target_shape)
        result_img = np.clip(result_img, 0, 1)
        result_img_display = (result_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(result_img_display)
        with st.session_state.intermediate_container:
            st.image(pil_img, caption=f"Generation {ga_instance.generations_completed}", width=200)

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    max_size = 100
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_height = int((height * max_size) / width)
            new_width = max_size
        else:
            new_width = int((width * max_size) / height)
            new_height = max_size
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return img_array

def create_fitness_plot(fitness_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=fitness_history,
        mode='lines',
        name='Fitness',
        line=dict(color='#667eea', width=3)
    ))
    fig.update_layout(
        title="Fitness Evolution Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness Value",
        template="plotly_white",
        height=400
    )
    return fig

def initialize_session_state():
    if 'target_image' not in st.session_state:
        st.session_state.target_image = None
    if 'target_chromosome' not in st.session_state:
        st.session_state.target_chromosome = None
    if 'target_shape' not in st.session_state:
        st.session_state.target_shape = None
    if 'fitness_history' not in st.session_state:
        st.session_state.fitness_history = []
    if 'ga_results' not in st.session_state:
        st.session_state.ga_results = None

def main():
    initialize_session_state()

    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🧬 Genetic Algorithm Image Reproducer
        </h1>
        <p style="color: white; text-align: center; margin-top: 0.5rem;">
            Upload an image and watch genetic algorithm evolve to reproduce it!
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Algorithm Parameters")
        num_generations = st.number_input("Number of Generations", min_value=100, max_value=50000, value=1000, step=100)
        population_size = st.number_input("Population Size", min_value=10, max_value=500, value=20, step=5)
        num_parents = st.number_input("Number of Parents", min_value=2, max_value=population_size, value=10, step=1)
        mutation_rate = st.number_input("Mutation Rate (%)", min_value=0.1, max_value=100.0, value=1.0,step=0.1)
        st.header("ℹ️ About")
        st.info("""
        This app uses genetic algorithm to evolve a random population of solutions 
        towards reproducing a target image.
        """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📁 Upload Target Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            st.session_state.target_image = processed_image
            st.session_state.target_chromosome = img2chromosome(processed_image)
            st.session_state.target_shape = processed_image.shape
            st.image(image, caption="Original Image", use_column_width=True)
            st.image(processed_image, caption=f"Processed Image {processed_image.shape}", use_column_width=True)
            st.success(f"""
            ✅ Image loaded successfully!
            - Dimensions: {processed_image.shape[1]}×{processed_image.shape[0]}
            - Channels: {processed_image.shape[2] if len(processed_image.shape) > 2 else 1}
            """)

    with col2:
        st.header("🚀 Evolution Results")
        if st.session_state.target_image is not None:
            if st.button("▶️ Start Evolution", type="primary", use_container_width=True):
                st.session_state.max_generations = num_generations
                evolution_col1, evolution_col2, evolution_col3 = st.columns([1, 1, 2])

                with evolution_col1:
                    st.subheader("📊 Evolution Progress")
                    progress_bar = st.progress(0)
                    st.session_state.progress_bar = progress_bar

                with evolution_col2:
                    st.subheader("📌 Metrics")
                    metrics_cols = st.columns(3)
                    st.session_state.metrics_cols = metrics_cols

                with evolution_col3:
                    st.subheader("🔄 Intermediate Results")
                    st.session_state.intermediate_container = st.empty()

                st.session_state.fitness_history = []

                ga_instance = pygad.GA(
                    num_generations=num_generations,
                    num_parents_mating=num_parents,
                    fitness_func=fitness_function,
                    sol_per_pop=population_size,
                    num_genes=st.session_state.target_image.size,
                    init_range_low=0.0,
                    init_range_high=1.0,
                    mutation_percent_genes=mutation_rate/100,
                    mutation_type="random",
                    mutation_by_replacement=True,
                    random_mutation_min_val=0.0,
                    random_mutation_max_val=1.0,
                    on_generation=on_generation_callback
                )

                with st.spinner("🧬 Evolution in progress..."):
                    start_time = time.time()
                    ga_instance.run()
                    end_time = time.time()

                st.session_state.ga_results = {
                    'ga_instance': ga_instance,
                    'execution_time': end_time - start_time
                }
                st.success(f"✅ Evolution completed in {end_time - start_time:.2f} seconds!")
        else:
            st.info("👆 Please upload an image first to start evolution.")

    if st.session_state.ga_results is not None:
        st.header("🎯 Final Results")
        ga_instance = st.session_state.ga_results['ga_instance']
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        result_img = chromosome2img(solution, st.session_state.target_shape)
        result_img = np.clip(result_img, 0, 1)
        result_img_display = (result_img * 255).astype(np.uint8)
        final_result = Image.fromarray(result_img_display)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🎯 Target Image")
            st.image(st.session_state.target_image, use_column_width=True)
        with col2:
            st.subheader("🧬 Evolved Image")
            st.image(final_result, use_column_width=True)
        with col3:
            st.subheader("📈 Evolution Stats")
            st.metric("Final Fitness", f"{solution_fitness:.2f}")
            st.metric("Best Generation", ga_instance.best_solution_generation)
            st.metric("Execution Time", f"{st.session_state.ga_results['execution_time']:.2f}s")

            img_buffer = io.BytesIO()
            final_result.save(img_buffer, format='PNG')
            st.download_button(
                label="📥 Download Result",
                data=img_buffer.getvalue(),
                file_name="genetic_algorithm_result.png",
                mime="image/png",
                use_container_width=True
            )

        if len(st.session_state.fitness_history) > 1:
            st.subheader("📊 Fitness Evolution")
            fitness_fig = create_fitness_plot(st.session_state.fitness_history)
            st.plotly_chart(fitness_fig, use_container_width=True)

if __name__ == "__main__":
    main()
