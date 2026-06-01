import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def gauge_chart(probability, risk_color):
    """Animated gauge showing default risk percentage."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Default Risk Score", 'font': {'size': 22}},
        delta={'reference': 50, 'increasing': {'color': "#E74C3C"},
               'decreasing': {'color': "#2ECC71"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1,
                     'tickcolor': "darkgray"},
            'bar': {'color': risk_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30],  'color': '#D5F5E3'},
                {'range': [30, 60], 'color': '#FDEBD0'},
                {'range': [60, 100],'color': '#FADBD8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        number={'suffix': "%", 'font': {'size': 36}}
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white"
    )
    return fig


def shap_waterfall(shap_df, top_n=8):
    """Horizontal waterfall bar chart showing feature contributions."""
    top = shap_df.head(top_n).copy()
    top = top.sort_values('SHAP Value')

    colors = ['#E74C3C' if v > 0 else '#2ECC71'
              for v in top['SHAP Value']]

    # Clean feature name labels
    labels = [f"{row['Feature']}<br><sub>value: {row['Raw Value']:.2f}</sub>"
              for _, row in top.iterrows()]

    fig = go.Figure(go.Bar(
        x=top['SHAP Value'],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top['SHAP Value']],
        textposition='outside'
    ))

    fig.update_layout(
        title='Why This Prediction? — Top Feature Contributions',
        xaxis_title='SHAP Value (impact on default risk)',
        plot_bgcolor='white',
        height=420,
        margin=dict(l=20, r=60, t=50, b=40),
        xaxis=dict(zeroline=True, zerolinecolor='black',
                   zerolinewidth=2)
    )
    return fig


def feature_importance_chart(model_metrics):
    """Bar chart of global feature importance."""
    fi = model_metrics['feature_importance']
    df = pd.DataFrame(list(fi.items()),
                      columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True)

    fig = px.bar(
        df, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Reds',
        title='Global Feature Importance — Random Forest'
    )
    fig.update_layout(
        plot_bgcolor='white', height=500,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig


def roc_placeholder():
    """Simple model performance summary chart."""
    metrics = ['AUC-ROC', 'Recall\n(Default)', 'Precision\n(No Default)']
    values  = [0.8496, 0.68, 0.97]
    colors  = ['#3498DB', '#E74C3C', '#2ECC71']

    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker_color=colors,
        text=[f'{v:.2f}' for v in values],
        textposition='outside'
    ))
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor='white',
        height=380
    )
    return fig