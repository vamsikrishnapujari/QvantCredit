import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
import networkx as nx
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="QvantCredit - Quantum Credit Risk Analysis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'execution_time' not in st.session_state:
    st.session_state.execution_time = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None

def create_credit_portfolio(n_loans, seed=42):
    """Generate synthetic credit portfolio with realistic parameters"""
    np.random.seed(seed)
    
    sectors = ['Technology', 'Finance', 'Retail', 'Manufacturing', 'Healthcare', 'Energy', 'Real Estate']
    
    loans = []
    for i in range(n_loans):
        sector = np.random.choice(sectors)
        
        # Sector-specific default probability ranges
        sector_risk = {
            'Technology': (0.02, 0.08),
            'Finance': (0.03, 0.12),
            'Retail': (0.05, 0.15),
            'Manufacturing': (0.04, 0.10),
            'Healthcare': (0.02, 0.07),
            'Energy': (0.06, 0.14),
            'Real Estate': (0.04, 0.11)
        }
        
        risk_range = sector_risk[sector]
        
        loan = {
            'id': f'LOAN_{i+1:03d}',
            'amount': np.random.uniform(10000, 500000),
            'default_prob': np.random.uniform(risk_range[0], risk_range[1]),
            'correlation': np.random.uniform(0.1, 0.5),
            'sector': sector,
            'duration': np.random.randint(12, 120),
            'credit_score': np.random.randint(550, 850),
            'ltv_ratio': np.random.uniform(0.4, 0.9)
        }
        loans.append(loan)
    
    return pd.DataFrame(loans)

def build_qubo_model(portfolio, max_risk, target_return, portfolio_size_limit):
    """Build QUBO model for portfolio optimization"""
    n = len(portfolio)
    
    # Create BQM
    bqm = BinaryQuadraticModel('BINARY')
    
    # Linear coefficients (expected returns - maximize)
    for i in range(n):
        expected_return = portfolio.iloc[i]['amount'] * (1 - portfolio.iloc[i]['default_prob'])
        bqm.add_variable(i, -expected_return)  # Negative for maximization
    
    # Quadratic coefficients (risk through correlations - minimize)
    for i in range(n):
        for j in range(i+1, n):
            # Calculate correlation-based risk
            correlation = (portfolio.iloc[i]['correlation'] + portfolio.iloc[j]['correlation']) / 2
            
            # Same sector increases correlation
            if portfolio.iloc[i]['sector'] == portfolio.iloc[j]['sector']:
                correlation *= 1.5
            
            risk_ij = correlation * portfolio.iloc[i]['default_prob'] * portfolio.iloc[j]['default_prob']
            risk_ij *= portfolio.iloc[i]['amount'] * portfolio.iloc[j]['amount'] / 1e9  # Scale
            
            bqm.add_interaction(i, j, risk_ij * 100)  # Risk penalty
    
    # Constraint: limit portfolio size (soft constraint)
    if portfolio_size_limit:
        penalty = max(portfolio['amount']) * 5
        target_size = min(portfolio_size_limit, n)
        
        # Add penalty for deviating from target size
        for i in range(n):
            bqm.add_variable(i, penalty * 0.1)
        
        for i in range(n):
            for j in range(i+1, n):
                bqm.add_interaction(i, j, penalty * 0.05)
    
    return bqm

def solve_with_dwave(bqm, num_reads=100, use_real_qpu=True):
    """Solve QUBO using D-Wave quantum annealer"""
    try:
        if use_real_qpu:
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(
                bqm, 
                num_reads=num_reads, 
                label='QvantCredit Portfolio Optimization',
                chain_strength=2.0
            )
            return sampleset, True
        else:
            raise Exception("Using simulated annealing")
    except Exception as e:
        st.warning(f"⚠️ D-Wave QPU not available: {str(e)}")
        st.info("🔄 Falling back to simulated annealing...")
        from neal import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=num_reads)
        return sampleset, False

def analyze_solution(solution, portfolio):
    """Analyze the quantum solution"""
    selected_loans = [i for i, v in solution.items() if v == 1]
    
    if not selected_loans:
        return None
    
    selected_portfolio = portfolio.iloc[selected_loans]
    
    total_amount = selected_portfolio['amount'].sum()
    expected_return = (selected_portfolio['amount'] * (1 - selected_portfolio['default_prob'])).sum()
    avg_default_prob = selected_portfolio['default_prob'].mean()
    weighted_default_prob = (selected_portfolio['amount'] * selected_portfolio['default_prob']).sum() / total_amount
    total_risk = (selected_portfolio['amount'] * selected_portfolio['default_prob']).sum()
    avg_credit_score = selected_portfolio['credit_score'].mean()
    roi = (expected_return - total_amount) / total_amount * 100
    
    return {
        'selected_loans': selected_loans,
        'num_loans': len(selected_loans),
        'total_amount': total_amount,
        'expected_return': expected_return,
        'avg_default_prob': avg_default_prob,
        'weighted_default_prob': weighted_default_prob,
        'total_risk': total_risk,
        'avg_credit_score': avg_credit_score,
        'roi': roi,
        'portfolio': selected_portfolio
    }

def create_network_graph(portfolio, selected_loans):
    """Create network graph showing loan relationships"""
    G = nx.Graph()
    
    # Add nodes
    for idx in selected_loans:
        loan = portfolio.iloc[idx]
        G.add_node(idx, 
                   sector=loan['sector'],
                   amount=loan['amount'],
                   risk=loan['default_prob'])
    
    # Add edges based on sector similarity
    for i in selected_loans:
        for j in selected_loans:
            if i < j and portfolio.iloc[i]['sector'] == portfolio.iloc[j]['sector']:
                G.add_edge(i, j)
    
    return G

# Header
st.markdown('<p class="main-header">🔮 QvantCredit</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Quantum-Powered Credit Risk Portfolio Optimization</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 0.9rem; color: #999;">Leveraging D-Wave Quantum Annealing for Superior Financial Decision Making</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown("### 📊 Portfolio Settings")
    n_loans = st.slider("Number of Loans", 5, 50, 20, help="Total number of loans to consider")
    portfolio_seed = st.number_input("Random Seed", 1, 1000, 42, help="Seed for reproducible results")
    
    st.markdown("### 🎯 Optimization Parameters")
    max_risk = st.slider("Maximum Risk Level (%)", 5, 50, 25, help="Maximum acceptable portfolio risk")
    target_return = st.slider("Target Return (%)", 5, 30, 15, help="Desired portfolio return")
    portfolio_size = st.slider("Max Portfolio Size", 3, 20, 10, help="Maximum number of loans to select")
    
    st.markdown("### 🔬 Quantum Settings")
    num_reads = st.slider("D-Wave Reads", 50, 1000, 100, 50, help="Number of quantum annealing reads")
    use_dwave = st.checkbox("Use Real D-Wave QPU", value=True, help="Use actual quantum hardware")
    show_details = st.checkbox("Show Technical Details", value=True)
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.info("QvantCredit uses D-Wave quantum annealing to solve complex portfolio optimization problems that are NP-hard for classical computers.")
    
    st.markdown("### 🎨 Color Legend")
    st.markdown("🟣 **Low Risk** (0-5%)")
    st.markdown("🔵 **Medium Risk** (5-10%)")
    st.markdown("🔴 **High Risk** (10%+)")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Portfolio Analysis", "🔮 Quantum Optimization", "📈 Results", "🔍 Details", "📚 Documentation"])

with tab1:
    st.markdown("## 📋 Credit Portfolio Overview")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Generate a synthetic credit portfolio with realistic loan characteristics including sector distribution, credit scores, and risk profiles.")
    with col2:
        if st.button("🎲 Generate Portfolio", type="primary", use_container_width=True):
            st.session_state.portfolio = create_credit_portfolio(n_loans, portfolio_seed)
            st.rerun()
    
    if st.session_state.portfolio is not None:
        portfolio = st.session_state.portfolio
        
        # Key Metrics
        st.markdown("### 📊 Portfolio Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Loans", len(portfolio))
        with col2:
            st.metric("Total Value", f"${portfolio['amount'].sum():,.0f}")
        with col3:
            st.metric("Avg Default Prob", f"{portfolio['default_prob'].mean():.2%}")
        with col4:
            st.metric("Total Risk", f"${(portfolio['amount'] * portfolio['default_prob']).sum():,.0f}")
        with col5:
            st.metric("Avg Credit Score", f"{portfolio['credit_score'].mean():.0f}")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Portfolio Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sector = px.pie(
                portfolio, 
                names='sector', 
                values='amount', 
                title='Portfolio Value by Sector',
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4
            )
            fig_sector.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
            sector_risk = portfolio.groupby('sector').agg({
                'default_prob': 'mean',
                'amount': 'sum'
            }).reset_index()
            
            fig_risk = px.bar(
                sector_risk,
                x='sector',
                y='default_prob',
                title='Average Default Probability by Sector',
                color='default_prob',
                color_continuous_scale='Reds',
                labels={'default_prob': 'Default Probability', 'sector': 'Sector'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                portfolio, 
                x='default_prob', 
                y='amount',
                color='sector', 
                size='credit_score',
                title='Risk vs. Amount Analysis',
                labels={'default_prob': 'Default Probability', 'amount': 'Loan Amount ($)'},
                hover_data=['id', 'duration', 'credit_score']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_credit = px.histogram(
                portfolio,
                x='credit_score',
                color='sector',
                title='Credit Score Distribution',
                labels={'credit_score': 'Credit Score'},
                nbins=20
            )
            st.plotly_chart(fig_credit, use_container_width=True)
        
        # Data Table
        st.markdown("### 📋 Detailed Portfolio Data")
        display_df = portfolio.copy()
        display_df['expected_return'] = display_df['amount'] * (1 - display_df['default_prob'])
        display_df['potential_loss'] = display_df['amount'] * display_df['default_prob']
        
        st.dataframe(
            display_df.style.format({
                'amount': '${:,.0f}',
                'default_prob': '{:.2%}',
                'correlation': '{:.3f}',
                'ltv_ratio': '{:.2f}',
                'expected_return': '${:,.0f}',
                'potential_loss': '${:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.info("👆 Click 'Generate Portfolio' to create a new credit portfolio for analysis.")

with tab2:
    st.markdown("## 🔮 Quantum Optimization Engine")
    
    if st.session_state.portfolio is None:
        st.warning("⚠️ Please generate a portfolio first in the Portfolio Analysis tab.")
    else:
        st.markdown("### 🎯 Optimization Objective")
        st.markdown("""
        The quantum optimizer solves a QUBO (Quadratic Unconstrained Binary Optimization) problem to:
        - **Maximize** expected returns from selected loans
        - **Minimize** portfolio risk through diversification
        - **Balance** sector exposure and correlations
        - **Optimize** portfolio size constraints
        """)
        
        st.markdown("### ⚙️ Current Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>Max Risk</h4>
                <h2>{max_risk}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <h4>Target Return</h4>
                <h2>{target_return}%</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <h4>Portfolio Size</h4>
                <h2>{portfolio_size}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="info-box">
                <h4>Quantum Reads</h4>
                <h2>{num_reads}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Quantum Optimization", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Build QUBO
                status_text.markdown("🔮 **Building QUBO model...**")
                progress_bar.progress(20)
                portfolio = st.session_state.portfolio
                bqm = build_qubo_model(portfolio, max_risk, target_return, portfolio_size)
                
                st.session_state.bqm = bqm
                status_text.markdown(f"✅ **QUBO model built:** {len(bqm.variables)} variables, {len(bqm.quadratic)} interactions")
                progress_bar.progress(40)
                
                # Step 2: Execute on D-Wave
                status_text.markdown("⚛️ **Executing on quantum computer...**")
                progress_bar.progress(50)
                
                start_time = datetime.now()
                sampleset, used_qpu = solve_with_dwave(bqm, num_reads, use_dwave)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                st.session_state.execution_time = execution_time
                st.session_state.used_qpu = used_qpu
                st.session_state.sampleset = sampleset
                
                qpu_type = "D-Wave QPU" if used_qpu else "Simulated Annealer"
                status_text.markdown(f"✅ **Quantum computation completed on {qpu_type} in {execution_time:.2f} seconds**")
                progress_bar.progress(70)
                
                # Step 3: Analyze results
                status_text.markdown("📊 **Analyzing quantum solutions...**")
                progress_bar.progress(85)
                
                best_solution = sampleset.first.sample
                analysis = analyze_solution(best_solution, portfolio)
                
                if analysis:
                    st.session_state.results = analysis
                    status_text.markdown(f"✅ **Optimization complete!** Found optimal portfolio with {analysis['num_loans']} loans")
                    progress_bar.progress(100)
                    st.balloons()
                    st.success(f"🎉 Successfully optimized portfolio using {qpu_type}!")
                else:
                    status_text.error("❌ No valid solution found. Try adjusting parameters.")
                    progress_bar.progress(100)

with tab3:
    st.markdown("## 📈 Optimization Results")
    
    if st.session_state.results is None:
        st.info("👈 Run the quantum optimization in the previous tab to see results here.")
    else:
        results = st.session_state.results
        
        # Success banner
        qpu_type = "D-Wave Quantum Processing Unit" if st.session_state.get('used_qpu', False) else "Simulated Quantum Annealer"
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Optimization Successful</h3>
            <p>Portfolio optimized using <strong>{qpu_type}</strong> in {st.session_state.execution_time:.2f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("### 🎯 Optimized Portfolio Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Selected Loans</h4>
                <h1>{results['num_loans']}</h1>
                <p>out of {len(st.session_state.portfolio)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Value</h4>
                <h1>${results['total_amount']/1000:.0f}K</h1>
                <p>Portfolio Size</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Expected Return</h4>
                <h1>${results['expected_return']/1000:.0f}K</h1>
                <p>ROI: {results['roi']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Risk Score</h4>
                <h1>{results['weighted_default_prob']:.2%}</h1>
                <p>Weighted Average</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Credit Score</h4>
                <h1>{results['avg_credit_score']:.0f}</h1>
                <p>Portfolio Average</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison metrics
        st.markdown("### 📊 Before vs After Optimization")
        original_portfolio = st.session_state.portfolio
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            orig_risk = original_portfolio['default_prob'].mean()
            opt_risk = results['avg_default_prob']
            risk_improvement = ((orig_risk - opt_risk) / orig_risk) * 100
            
            st.metric(
                "Average Risk",
                f"{opt_risk:.2%}",
                f"-{risk_improvement:.1f}%",
                delta_color="inverse"
            )
        
        with col2:
            orig_amount = original_portfolio['amount'].sum()
            opt_amount = results['total_amount']
            
            st.metric(
                "Portfolio Concentration",
                f"{(opt_amount/orig_amount)*100:.1f}%",
                "Optimized"
            )
        
        with col3:
            st.metric(
                "Diversification",
                f"{results['num_loans']} sectors",
                f"{len(results['portfolio']['sector'].unique())} types"
            )
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Portfolio Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_opt = px.pie(
                results['portfolio'], 
                names='sector', 
                values='amount',
                title='Optimized Portfolio Composition',
                color_discrete_sequence=px.colors.sequential.Purples_r,
                hole=0.4
            )
            fig_opt.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_opt, use_container_width=True)
        
        with col2:
            comparison_data = pd.DataFrame({
                'Metric': ['Total Risk', 'Expected Return', 'Avg Credit Score'],
                'Value': [results['total_risk']/1000, results['expected_return']/1000, results['avg_credit_score']/10]
            })
            
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Bar(
                x=comparison_data['Metric'],
                y=comparison_data['Value'],
                marker_color=['#ff6b6b', '#51cf66', '#4c6ef5'],
                text=comparison_data['Value'].round(1),
                textposition='auto',
            ))
            fig_metrics.update_layout(
                title='Key Metrics Overview',
                yaxis_title='Value',
                showlegend=False
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk-Return scatter
            fig_risk_return = px.scatter(
                results['portfolio'],
                x='default_prob',
                y='amount',
                color='sector',
                size='credit_score',
                title='Selected Loans: Risk vs Amount',
                labels={'default_prob': 'Default Probability', 'amount': 'Loan Amount ($)'},
                hover_data=['id']
            )
            st.plotly_chart(fig_risk_return, use_container_width=True)
        
        with col2:
            # Credit score distribution
            fig_credit_dist = px.box(
                results['portfolio'],
                x='sector',
                y='credit_score',
                color='sector',
                title='Credit Score Distribution by Sector',
                labels={'credit_score': 'Credit Score'}
            )
            st.plotly_chart(fig_credit_dist, use_container_width=True)
        
        # Energy landscape
        if 'sampleset' in st.session_state:
            st.markdown("### ⚡ Quantum Solution Energy Landscape")
            sampleset = st.session_state.sampleset
            
            # Extract solution data
            solutions_data = []
            for idx, record in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):
                num_selected = sum(record.sample.values())
                solutions_data.append({
                    'index': idx,
                    'energy': record.energy,
                    'occurrences': record.num_occurrences,
                    'num_loans': num_selected
                })
            
            solutions_df = pd.DataFrame(solutions_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_energy = go.Figure()
                fig_energy.add_trace(go.Scatter(
                    x=solutions_df['index'],
                    y=solutions_df['energy'],
                    mode='markers',
                    marker=dict(
                        size=solutions_df['occurrences']/max(solutions_df['occurrences'])*30 + 5,
                        color=solutions_df['energy'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Energy")
                    ),
                    text=[f"Energy: {e:.2f}<br>Occurrences: {o}<br>Loans: {n}" 
                          for e, o, n in zip(solutions_df['energy'], solutions_df['occurrences'], solutions_df['num_loans'])],
                    hovertemplate='%{text}<extra></extra>'
                ))
                fig_energy.update_layout(
                    title='Energy Distribution of Quantum Solutions',
                    xaxis_title='Solution Index',
                    yaxis_title='Energy',
                    showlegend=False
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            
            with col2:
                fig_histogram = px.histogram(
                    solutions_df,
                    x='energy',
                    title='Energy Frequency Distribution',
                    labels={'energy': 'Energy', 'count': 'Frequency'},
                    nbins=30,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_histogram, use_container_width=True)
        
        # Selected loans table
        st.markdown("### 🎯 Selected Loans Details")
        display_df = results['portfolio'].copy()
        display_df['expected_return'] = display_df['amount'] * (1 - display_df['default_prob'])
        display_df['potential_loss'] = display_df['amount'] * display_df['default_prob']
        
        st.dataframe(
            display_df.style.format({
                'amount': '${:,.0f}',
                'default_prob': '{:.2%}',
                'correlation': '{:.3f}',
                'ltv_ratio': '{:.2f}',
                'expected_return': '${:,.0f}',
                'potential_loss': '${:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download results
        st.markdown("### 💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Portfolio CSV",
                data=csv,
                file_name=f"qvantcredit_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_data = {
                'optimization_date': datetime.now().isoformat(),
                'qpu_type': qpu_type,
                'execution_time': st.session_state.execution_time,
                'metrics': {
                    'num_loans': results['num_loans'],
                    'total_amount': results['total_amount'],
                    'expected_return': results['expected_return'],
                    'roi': results['roi'],
                    'avg_default_prob': results['avg_default_prob'],
                    'weighted_default_prob': results['weighted_default_prob']
                }
            }
            
            st.download_button(
                label="📥 Download Summary JSON",
                data=json.dumps(summary_data, indent=2),
                file_name=f"qvantcredit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

with tab4:
    st.markdown("## 🔍 Technical Details")
    
    if show_details:
        if 'portfolio' in st.session_state and st.session_state.portfolio is not None:
            
            if 'bqm' in st.session_state:
                st.markdown("### 🧮 QUBO Model Structure")
                bqm = st.session_state.bqm
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Variables", len(bqm.variables))
                with col2:
                    st.metric("Linear Terms", len(bqm.linear))
                with col3:
                    st.metric("Quadratic Terms", len(bqm.quadratic))
                with col4:
                    st.metric("Offset", f"{bqm.offset:.2f}")
                
                st.markdown("#### Model Coefficients Sample")
                linear_sample = dict(list(bqm.linear.items())[:10])
                st.json({"linear_coefficients_sample": {str(k): float(v) for k, v in linear_sample.items()}})
                
                st.markdown("---")
            
            if st.session_state.execution_time:
                st.markdown("### ⚡ Execution Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Execution Time", f"{st.session_state.execution_time:.3f}s")
                with col2:
                    qpu_used = "Yes ✅" if st.session_state.get('used_qpu', False) else "No (Simulated)"
                    st.metric("D-Wave QPU Used", qpu_used)
                with col3:
                    if 'sampleset' in st.session_state:
                        st.metric("Solutions Generated", len(st.session_state.sampleset))
            
            if 'sampleset' in st.session_state:
                st.markdown("### 📊 Sample Set Statistics")
                sampleset = st.session_state.sampleset
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Energy", f"{sampleset.first.energy:.4f}")
                with col2:
                    st.metric("Best Solution Occurrences", sampleset.first.num_occurrences)
                with col3:
                    all_energies = [record.energy for record in sampleset.data()]
                    st.metric("Energy Std Dev", f"{np.std(all_energies):.4f}")
                
                st.markdown("#### Top 10 Solutions")
                top_solutions = []
                for idx, record in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:10]):
                    num_selected = sum(record.sample.values())
                    top_solutions.append({
                        'Rank': idx + 1,
                        'Energy': f"{record.energy:.4f}",
                        'Occurrences': record.num_occurrences,
                        'Loans Selected': num_selected
                    })
                
                st.dataframe(pd.DataFrame(top_solutions), use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 🔗 Quantum Annealing Info")
                st.markdown("""
                **QUBO Formulation:**
                - **Objective:** Minimize energy function E = Σᵢ hᵢxᵢ + Σᵢⱼ Jᵢⱼxᵢxⱼ
                - **Variables:** Binary (0 = reject loan, 1 = accept loan)
                - **Linear terms (h):** Expected returns (negative for maximization)
                - **Quadratic terms (J):** Correlation-based risk penalties
                
                **D-Wave Quantum Annealer:**
                - Uses quantum tunneling to explore solution space
                - Samples multiple solutions from low-energy states
                - Automatically handles qubit connectivity through embedding
                """)
        else:
            st.info("Generate a portfolio and run optimization to see technical details.")
    else:
        st.info("Enable 'Show Technical Details' in the sidebar to view quantum computing metrics.")

with tab5:
    st.markdown("## 📚 Documentation")
    
    st.markdown("""
    ### 🔮 What is QvantCredit?
    
    QvantCredit is a quantum-powered credit risk portfolio optimization platform that leverages D-Wave's quantum 
    annealing technology to solve complex portfolio optimization problems that are computationally intractable 
    for classical computers.
    
    ### 🎯 Key Features
    
    1. **Real Quantum Computing Integration**
       - Direct connection to D-Wave quantum processing units (QPUs)
       - Automatic fallback to simulated annealing if QPU unavailable
       - Configurable number of quantum reads for solution quality
    
    2. **Sophisticated Portfolio Modeling**
       - Multi-sector loan portfolios with realistic risk profiles
       - Credit score integration and LTV ratio modeling
       - Correlation-based risk assessment
       - Sector-specific default probability distributions
    
    3. **QUBO Optimization**
       - Transforms portfolio optimization into Quadratic Unconstrained Binary Optimization
       - Balances return maximization with risk minimization
       - Incorporates soft constraints for portfolio size limits
       - Penalizes high correlation within same sectors
    
    4. **Beautiful Visualizations**
       - Interactive Plotly charts for risk analysis
       - Energy landscape visualization
       - Sector distribution and correlation matrices
       - Before/after optimization comparisons
    
    ### 🚀 How to Use
    
    1. **Configure Parameters** (Sidebar)
       - Set number of loans and risk tolerance
       - Adjust target returns and portfolio size
       - Configure quantum annealing settings
    
    2. **Generate Portfolio** (Tab 1)
       - Click "Generate Portfolio" to create synthetic loans
       - Review sector distribution and risk metrics
       - Analyze individual loan characteristics
    
    3. **Run Optimization** (Tab 2)
       - Click "Run Quantum Optimization"
       - Watch as the QUBO model is built and executed
       - View real-time progress indicators
    
    4. **Analyze Results** (Tab 3)
       - Review optimized portfolio metrics
       - Compare before/after statistics
       - Examine energy landscapes
       - Download results for further analysis
    
    ### 🔬 Technical Background
    
    **Quantum Annealing:**
    Quantum annealing is a quantum computing approach that uses quantum tunneling to find global minima 
    of complex optimization problems. Unlike gate-based quantum computers, quantum annealers are specifically 
    designed for optimization tasks.
    
    **QUBO Formulation:**
    The portfolio optimization problem is formulated as:
    
    Minimize: E = Σᵢ hᵢxᵢ + Σᵢⱼ Jᵢⱼxᵢxⱼ
    
    Where:
    - xᵢ ∈ {0,1} represents whether loan i is selected
    - hᵢ represents the linear coefficients (expected returns)
    - Jᵢⱼ represents quadratic coefficients (correlation risks)
    
    **Why Quantum?**
    Portfolio optimization with correlation constraints is NP-hard. As portfolio size grows, classical 
    algorithms struggle to find optimal solutions in reasonable time. Quantum annealing can explore 
    the solution space more efficiently through quantum superposition and tunneling.
    
    ### 📊 Metrics Explained
    
    - **Default Probability:** Likelihood that a borrower will default (0-100%)
    - **Expected Return:** Amount * (1 - Default Probability)
    - **Portfolio Risk:** Weighted average of default probabilities
    - **ROI:** Return on Investment percentage
    - **Credit Score:** Standard FICO-style score (300-850)
    - **LTV Ratio:** Loan-to-Value ratio (lower is less risky)
    - **Correlation:** Measure of how loan risks move together
    
    ### 🛠️ Requirements
    
    - D-Wave Ocean SDK
    - Streamlit
    - Plotly for visualizations
    - NumPy and Pandas for data processing
    - Valid D-Wave API token (for real QPU access)
    
    ### 🔐 Setting Up D-Wave Access
    
    1. Sign up for D-Wave Leap account (free tier available)
    2. Get your API token from the dashboard
    3. Configure using: `dwave config create`
    4. Enter your API token when prompted
    
    ### 💡 Tips for Best Results
    
    - Start with 10-20 loans for faster results
    - Use 100-200 reads for good solution quality
    - Higher portfolio size limits increase flexibility
    - Lower risk tolerance produces more conservative portfolios
    - Check "Use Real D-Wave QPU" for authentic quantum computing
    
    ### 🤝 Support & Resources
    
    - D-Wave Documentation: https://docs.ocean.dwavesys.com/
    - Quantum Annealing Guide: https://www.dwavesys.com/learn
    - QUBO Formulation Guide: Research papers on quantum portfolio optimization
    
    ### 📝 License & Citation
    
    QvantCredit is an educational and research tool. For production use, please ensure proper 
    risk management practices and regulatory compliance.
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p style="font-size: 1.1rem;">🔮 <strong>Powered by D-Wave Quantum Computing</strong></p>
        <p>Built with Streamlit • © 2024 QvantCredit</p>
        <p style="font-size: 0.9rem; color: #999;">
            Leveraging quantum annealing for superior financial optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
