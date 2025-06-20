#CALCULATE BOILER EFFICIENCY AND AIR PREHEATER ANALYSIS STREAMLIT DASHBOARD

import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import math

# Set page configuration
st.set_page_config(layout="wide", page_title="Air Preheater & Boiler Efficiency Dashboard")

# --- Logo and Tabs ---
try:
    tce_logo = Image.open('tce_logo.jpg')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(tce_logo, width=700)
except:
    st.warning("TCE logo not found. Please ensure 'tce_logo.jpg' is in the same directory.")

tab1, tab2 = st.tabs(["Air Preheater Analysis", "Boiler Efficiency"])

# --- Functions for Air Preheater Analysis ---
def load_airpre_data(file_path):
    """Load and process air preheater data"""
    df = pd.read_excel(file_path, usecols=["Description", "Design", "Actual"])
    
    parameters = [
        'RH', 'Atmospheric pressure', 'Saturation Vapor pressure of moisture',
        'Mass fraction of water vapor in dry air', 'Mass fraction of water vapor in wet air',
        'Avg. Flue Gas O2 - APH In', 'Avg. Flue Gas O2 - APH Out',
        'Avg. Flue Gas Temp - APH In', 'Avg. Flue Gas Temp - APH Out',
        'Primary Air to APH Temp In', 'Secondary Air to APH Temp In',
        'Primary Air to APH Temp out', 'Secondary Air to APH Temp out',
        'Total Secondary Air Flow', 'Total Primary Air Flow', 'Dry bulb Temp'
    ]
    
    results = {}
    for param in parameters:
        row = df[df['Description'].str.contains(param, na=False)]
        if not row.empty:
            results[param] = {
                'Design': row['Design'].iloc[0],
                'Actual': row['Actual'].iloc[0]
            }
    return results

def calculate_aph_metrics(results):
    """Calculate air preheater performance metrics"""
    # APH Leakage calculation
    o2_in_design = results['Avg. Flue Gas O2 - APH In']['Design']
    o2_out_design = results['Avg. Flue Gas O2 - APH Out']['Design']
    o2_in_actual = results['Avg. Flue Gas O2 - APH In']['Actual']
    o2_out_actual = results['Avg. Flue Gas O2 - APH Out']['Actual']
    
    aph_leakage_design = ((o2_out_design - o2_in_design)/(21 - o2_out_design)) * 90
    aph_leakage_actual = ((o2_out_actual - o2_in_actual)/(21 - o2_out_actual)) * 90
    
    # Air flow calculations
    total_air_design = results['Total Primary Air Flow']['Design'] + results['Total Secondary Air Flow']['Design']
    total_air_actual = results['Total Primary Air Flow']['Actual'] + results['Total Secondary Air Flow']['Actual']
    
    # Air flow ratios
    sec_air_ratio_design = results['Total Secondary Air Flow']['Design'] / total_air_design 
    sec_air_ratio_actual = results['Total Secondary Air Flow']['Actual'] / total_air_actual
    
    pri_air_ratio_design = results['Total Primary Air Flow']['Design'] / total_air_design
    pri_air_ratio_actual = results['Total Primary Air Flow']['Actual'] / total_air_actual
    
    # Temperature calculations
    aph_inlet_temp_design = (sec_air_ratio_design * results['Secondary Air to APH Temp In']['Design'] + 
                            pri_air_ratio_design * results['Primary Air to APH Temp In']['Design']) / (sec_air_ratio_design + pri_air_ratio_design)
    aph_inlet_temp_actual = (sec_air_ratio_actual * results['Secondary Air to APH Temp In']['Actual'] + 
                            pri_air_ratio_actual * results['Primary Air to APH Temp In']['Actual']) / (sec_air_ratio_actual + pri_air_ratio_actual)
    
    aph_outlet_temp_design = (sec_air_ratio_design * results['Secondary Air to APH Temp out']['Design'] + 
                             pri_air_ratio_design * results['Primary Air to APH Temp out']['Design']) / (sec_air_ratio_design + pri_air_ratio_design)
    aph_outlet_temp_actual = (sec_air_ratio_actual * results['Secondary Air to APH Temp out']['Actual'] + 
                             pri_air_ratio_actual * results['Primary Air to APH Temp out']['Actual']) / (sec_air_ratio_actual + pri_air_ratio_actual)
    
    # CP calculations
    def convert_c_to_f(celsius):
        return (celsius * 9/5) + 32
    
    def calculate_cp_air(temp_f):
        term1 = -0.000000000005 * (2 * temp_f - 77)**3
        term2 = 0.00000001 * (2 * temp_f - 77)**2
        term3 = 0.0000002 * (2 * temp_f - 77)
        term4 = 0.24
        return (term1 + term2 + term3 + term4) * 4.184
    
    def calculate_cp_flue_gas(temp_c):
        temp_f = convert_c_to_f(temp_c)
        cp1 = (0.00002 * (2 * temp_f - 77) + 0.2343) * 4.184
        cp2 = (0.00002 * (2 * convert_c_to_f(temp_c + 5) - 77) + 0.2343) * 4.184
        return (cp1 + cp2) / 2
    
    # Calculate CP values
    temp1_f_design = convert_c_to_f(aph_inlet_temp_design)
    temp2_f_design = convert_c_to_f(results['Primary Air to APH Temp In']['Design'])
    cp_air_design = (calculate_cp_air(temp1_f_design) + calculate_cp_air(temp2_f_design)) / 2
    
    temp1_f_actual = convert_c_to_f(aph_inlet_temp_actual)
    temp2_f_actual = convert_c_to_f(results['Primary Air to APH Temp In']['Actual'])
    cp_air_actual = (calculate_cp_air(temp1_f_actual) + calculate_cp_air(temp2_f_actual)) / 2
    
    cp_flue_gas_design = calculate_cp_flue_gas(results['Avg. Flue Gas Temp - APH Out']['Design'])
    cp_flue_gas_actual = calculate_cp_flue_gas(results['Avg. Flue Gas Temp - APH Out']['Actual'])
    
    # Calculate corrected APH exit temperature
    flue_gas_exit_temp_design = (aph_leakage_design * cp_air_design * (results['Avg. Flue Gas Temp - APH Out']['Design'] - aph_inlet_temp_design) / 
                                (cp_flue_gas_design * 100)) + results['Avg. Flue Gas Temp - APH Out']['Design']
    flue_gas_exit_temp_actual = (aph_leakage_actual * cp_air_actual * (results['Avg. Flue Gas Temp - APH Out']['Actual'] - aph_inlet_temp_actual) / 
                                (cp_flue_gas_actual * 100)) + results['Avg. Flue Gas Temp - APH Out']['Actual']
    
    # Calculate efficiencies
    gas_side_eff_design = ((results['Avg. Flue Gas Temp - APH In']['Design'] - flue_gas_exit_temp_design) * 100 / 
                          (results['Avg. Flue Gas Temp - APH In']['Design'] - aph_inlet_temp_design))
    gas_side_eff_actual = ((results['Avg. Flue Gas Temp - APH In']['Actual'] - flue_gas_exit_temp_actual) * 100 / 
                          (results['Avg. Flue Gas Temp - APH In']['Actual'] - aph_inlet_temp_actual))
    
    # Calculate X ratio
    x_ratio_design = (results['Avg. Flue Gas Temp - APH In']['Design'] - flue_gas_exit_temp_design) / (aph_outlet_temp_design - aph_inlet_temp_design)
    x_ratio_actual = (results['Avg. Flue Gas Temp - APH In']['Actual'] - flue_gas_exit_temp_actual) / (aph_outlet_temp_actual - aph_inlet_temp_actual)
    
    # Return all metrics
    return {
        'APH Leakage': {'Design': aph_leakage_design, 'Actual': aph_leakage_actual},
        'Total Air Flow': {'Design': total_air_design, 'Actual': total_air_actual},
        'Secondary Air Flow Ratio': {'Design': sec_air_ratio_design, 'Actual': sec_air_ratio_actual},
        'Primary Air Flow Ratio': {'Design': pri_air_ratio_design, 'Actual': pri_air_ratio_actual},
        'APH Inlet Temperature Average': {'Design': aph_inlet_temp_design, 'Actual': aph_inlet_temp_actual},
        'APH Outlet Temperature Average': {'Design': aph_outlet_temp_design, 'Actual': aph_outlet_temp_actual},
        'CP of Air': {'Design': cp_air_design, 'Actual': cp_air_actual},
        'CP of Flue Gas': {'Design': cp_flue_gas_design, 'Actual': cp_flue_gas_actual},
        'Corrected APH Exit Temperature': {'Design': flue_gas_exit_temp_design, 'Actual': flue_gas_exit_temp_actual},
        'Gas Side Efficiency': {'Design': gas_side_eff_design, 'Actual': gas_side_eff_actual},
        'X Ratio': {'Design': x_ratio_design, 'Actual': x_ratio_actual}
    }
# --- Tab 1: Air Preheater Analysis ---
with tab1:
    st.title("Air Preheater Analysis Dashboard")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Air Preheater Data (Excel)", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load and process data
            results = load_airpre_data(uploaded_file)
            metrics = calculate_aph_metrics(results)
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Design Parameters")
                for metric, values in metrics.items():
                    st.metric(
                        label=metric,
                        value=f"{values['Design']:.2f}",
                        delta=None
                    )
            
            with col2:
                st.subheader("Actual Parameters")
                for metric, values in metrics.items():
                    st.metric(
                        label=metric,
                        value=f"{values['Actual']:.2f}",
                        delta=f"{values['Actual'] - values['Design']:.2f}"
                    )
            
            # Add visualization
            st.subheader("Performance Comparison")
            
            # Create comparison chart
            fig = go.Figure()
            
            for metric, values in metrics.items():
                fig.add_trace(go.Bar(
                    name=metric,
                    x=['Design', 'Actual'],
                    y=[values['Design'], values['Actual']],
                    text=[f"{values['Design']:.2f}", f"{values['Actual']:.2f}"],
                    textposition='auto',
                ))
            
            fig.update_layout(
                barmode='group',
                title="Design vs Actual Performance Metrics",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add APH-specific visualization
            st.subheader("Air Preheater Key Parameters")
            
            # Create a custom figure for APH metrics
            fig_aph = go.Figure()
            
            # Temperature comparison
            fig_aph.add_trace(go.Indicator(
                mode="number+gauge+delta",
                value=metrics['Corrected APH Exit Temperature']['Actual'],
                title={'text': "APH Exit Temperature (°C)"},
                delta={'reference': metrics['Corrected APH Exit Temperature']['Design']},
                gauge={
                    'axis': {'range': [0, 200]},
                    'bar': {'color': "orange"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics['Corrected APH Exit Temperature']['Design']
                    }
                },
                domain={'row': 0, 'column': 0}
            ))

            # APH Leakage
            fig_aph.add_trace(go.Indicator(
                mode="number+gauge+delta",
                value=metrics['APH Leakage']['Actual'],
                title={'text': "APH Leakage (%)"},
                delta={'reference': metrics['APH Leakage']['Design']},
                gauge={
                    'axis': {'range': [0, 5]},
                    'bar': {'color': "blue"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics['APH Leakage']['Design']
                    }
                },
                domain={'row': 0, 'column': 1}
            ))

            # Gas Side Efficiency
            fig_aph.add_trace(go.Indicator(
                mode="number+gauge+delta",
                value=metrics['Gas Side Efficiency']['Actual'],
                title={'text': "Gas Side Efficiency (%)"},
                delta={'reference': metrics['Gas Side Efficiency']['Design']},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics['Gas Side Efficiency']['Design']
                    }
                },
                domain={'row': 1, 'column': 0}
            ))

            # X-Ratio
            fig_aph.add_trace(go.Indicator(
                mode="number+gauge+delta",
                value=metrics['X Ratio']['Actual'],
                title={'text': "X-Ratio"},
                delta={'reference': metrics['X Ratio']['Design']},
                gauge={
                    'axis': {'range': [0, 2]},
                    'bar': {'color': "purple"},
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': metrics['X Ratio']['Design']
                    }
                },
                domain={'row': 1, 'column': 1}
            ))

            # Update layout for 2x2 grid
            fig_aph.update_layout(
                grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
                height=800
            )

            st.plotly_chart(fig_aph, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.warning("Please check your input file format")
    
    else:
        st.info("Please upload an Excel file containing Air Preheater data")

# --- Tab 2: Boiler Efficiency ---
def calculate_boiler_efficiency_from_losses(losses):
    """
    Calculate boiler efficiency from given loss components
    losses: dictionary containing loss values for each category
    """
    total_loss = sum(losses.values())
    efficiency = 100 - total_loss
    return efficiency, total_loss

with tab2:
    st.title("Boiler Efficiency Calculation (PTC 4 Standard)")
    st.markdown("---")

    try:
        # Define losses as per the table
        design_losses = {
            'Dry Gas Loss': 5.354,
            'Loss due to Unburnt Carbon': 0.502,
            'Loss due to moisture in fuel': 1.113,
            'Loss due to Hydrogen in Fuel': 4.295,
            'Loss due to moisture in air': 0.134,
            'Loss due to Radiation & Unaccounted': 0.940
        }

        derived_losses = {
            'Dry Gas Loss': 5.194,
            'Loss due to Unburnt Carbon': 0.499,
            'Loss due to moisture in fuel': 1.106,
            'Loss due to Hydrogen in Fuel': 3.757,
            'Loss due to moisture in air': 0.129,
            'Loss due to Radiation & Unaccounted': 0.940
        }

        actual_losses = {
            'Dry Gas Loss': 5.354,
            'Loss due to Unburnt Carbon': 0.502,
            'Loss due to moisture in fuel': 1.113,
            'Loss due to Hydrogen in Fuel': 4.295,
            'Loss due to moisture in air': 0.134,
            'Loss due to Radiation & Unaccounted': 0.940
        }

        # Calculate efficiencies
        efficiency_design, total_loss_design = calculate_boiler_efficiency_from_losses(design_losses)
        efficiency_derived, total_loss_derived = calculate_boiler_efficiency_from_losses(derived_losses)
        efficiency_actual, total_loss_actual = calculate_boiler_efficiency_from_losses(actual_losses)

        # Display results in three columns
        col1, col2, col3 = st.columns(3)

        with col1:
           #st.subheader("Design Values")
            st.markdown("<h3 style='text-align: center;'>Design Values</h3>", unsafe_allow_html=True)
            gauge_fig_design = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efficiency_design,
                title={'text': "Design Efficiency", 'align': 'center'},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#0070c0"},
                    'steps': [
                        {'range': [0, 70], 'color': "#e0e0e0"},
                        {'range': [70, 90], 'color': "#b3d1f2"},
                        {'range': [90, 100], 'color': "#d2f2b3"}
                    ]
                }
            ))
            st.plotly_chart(gauge_fig_design, use_container_width=True)
            st.metric("Total Loss", f"{total_loss_design:.3f}%")
            
        with col2:
            st.markdown("<h3 style='text-align: center;'>Derived Values</h3>", unsafe_allow_html=True)
            gauge_fig_derived = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efficiency_derived,
                title={'text': "Derived Efficiency"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#28a745"},
                    'steps': [
                        {'range': [0, 70], 'color': "#e0e0e0"},
                        {'range': [70, 90], 'color': "#b3d1f2"},
                        {'range': [90, 100], 'color': "#d2f2b3"}
                    ]
                }
            ))
            st.plotly_chart(gauge_fig_derived, use_container_width=True)
            st.metric("Total Loss", f"{total_loss_derived:.3f}%")

        with col3:
            st.markdown("<h3 style='text-align: center;'>Actual Values</h3>", unsafe_allow_html=True)
            gauge_fig_actual = go.Figure(go.Indicator(
                mode="gauge+number",
                value=efficiency_actual,
                title={'text': "Actual Efficiency"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#dc3545"},
                    'steps': [
                        {'range': [0, 70], 'color': "#e0e0e0"},
                        {'range': [70, 90], 'color': "#b3d1f2"},
                        {'range': [90, 100], 'color': "#d2f2b3"}
                    ]
                }
            ))
            st.plotly_chart(gauge_fig_actual, use_container_width=True)
            st.metric("Total Loss", f"{total_loss_actual:.3f}%")

        # Display detailed comparison table
        st.markdown("### Detailed Loss Components")
        comparison_data = pd.DataFrame({
            'Loss Component': design_losses.keys(),
            'Design (%)': design_losses.values(),
            'Derived (%)': derived_losses.values(),
            'Actual (%)': actual_losses.values()
        })
        st.table(comparison_data.round(3))

        # Add summary metrics
        st.markdown("### Summary")
        st.markdown(f"""
        | Parameter | Design | Derived | Actual |
        |-----------|---------|----------|---------|
        | Total Loss (%) | {total_loss_design:.3f} | {total_loss_derived:.3f} | {total_loss_actual:.3f} |
        | Boiler Efficiency (%) | {efficiency_design:.3f} | {efficiency_derived:.3f} | {efficiency_actual:.3f} |
        """)

    except Exception as e:
        st.error(f"Error in calculations: {str(e)}")

    st.markdown("---")
    st.markdown("*Boiler Efficiency calculation as per PTC 4 Standard*")

    # --- Boiler Efficiency Calculation Function ---
    def calculate_boiler_efficiency(excel_path):
        """
        Calculates both DESIGN and OPERATING boiler efficiency from the given Excel file.
        Returns a dictionary with all loss components and efficiencies.
        """
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip()
        df['PARAMETERS'] = df['PARAMETERS'].str.strip()

        def get_param(name, col):
            return float(df.loc[df['PARAMETERS'] == name, col].values[0])

        # Constants
        HStLvCr = 2771.358515
        HWRe = 104.67
        MFrWH2F = 0.373790639
        L1 = 5.194121064
        L2 = 0.498607275
        L5 = 0.128841522
        L6 = 0.94

        # DESIGN values
        M_design = get_param('MOISTURE CONTENT IN COAL', 'DESIGN')
        GCV_design = get_param('GCV OF COAL', 'DESIGN')
        L3_design = (M_design * (HStLvCr - HWRe)) / (GCV_design * 4.184)
        L4_design = (MFrWH2F * (HStLvCr - HWRe)) / (GCV_design * 4.184)
        total_loss_design = L1 + L2 + L3_design + L4_design + L5 + L6
        efficiency_design = 100 - total_loss_design

        # OPERATING values
        M_oper = get_param('MOISTURE CONTENT IN COAL', 'OPERATING')
        GCV_oper = get_param('GCV OF COAL', 'OPERATING')
        L3_oper = (M_oper * (HStLvCr - HWRe)) / (GCV_oper * 4.184)
        L4_oper = (MFrWH2F * (HStLvCr - HWRe)) / (GCV_oper * 4.184)
        total_loss_oper = L1 + L2 + L3_oper + L4_oper + L5 + L6
        efficiency_oper = 100 - total_loss_oper

        return {
            "design": {
                "L1": L1,
                "L2": L2,
                "L3": L3_design,
                "L4": L4_design,
                "L5": L5,
                "L6": L6,
                "total_loss": total_loss_design,
                "efficiency": efficiency_design
            },
            "operating": {
                "L1": L1,
                "L2": L2,
                "L3": L3_oper,
                "L4": L4_oper,
                "L5": L5,
                "L6": L6,
                "total_loss": total_loss_oper,
                "efficiency": efficiency_oper
            }
        }

    # Example usage:
    # result = calculate_boiler_efficiency('binput1.xlsx')
    # print(f"Design Efficiency: {result['design']['efficiency']:.4f}%")
    # print(f"Operating Efficiency: {result['operating']['efficiency']:.4f}%")

    # Create Waterfall and Pie Charts for Loss Distribution
    st.subheader("Heat Loss Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Waterfall Chart
        measure = ['absolute'] + ['relative'] * len(actual_losses) + ['total']
        x_data = ['Input Energy'] + list(actual_losses.keys()) + ['Useful Heat']
        y_data = [100] + [-val for val in actual_losses.values()] + [efficiency_actual]
        
        waterfall = go.Figure(go.Waterfall(
            name="Loss Distribution",
            orientation="v",
            measure=measure,
            x=x_data,
            y=y_data,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        waterfall.update_layout(
            title="Energy Flow Distribution",
            showlegend=False,
            height=600,
            xaxis=dict(tickangle=45),
            yaxis_title="Percentage (%)"
        )
        
        st.plotly_chart(waterfall, use_container_width=True)

    with col2:
        # Pie Chart
        pie_labels = list(actual_losses.keys()) + ['Useful Heat']
        pie_values = list(actual_losses.values()) + [efficiency_actual]
        
        pie = go.Figure(data=[go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=.4,
            marker_colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#90EE90']
        )])
        
        pie.update_layout(
            title="Loss Components Distribution",
            height=600,
            annotations=[dict(text=f'Total\nEfficiency\n{efficiency_actual:.1f}%', x=0.5, y=0.5, font_size=15, showarrow=False)]
        )
        
        st.plotly_chart(pie, use_container_width=True)

    # Add a bar chart comparison
    st.subheader("Loss Components Comparison")
    bar_fig = go.Figure(data=[
        go.Bar(name='Design', x=list(design_losses.keys()), y=list(design_losses.values())),
        go.Bar(name='Derived', x=list(derived_losses.keys()), y=list(derived_losses.values())),
        go.Bar(name='Actual', x=list(actual_losses.keys()), y=list(actual_losses.values()))
    ])

    bar_fig.update_layout(
        barmode='group',
        title="Design vs Derived vs Actual Losses",
        xaxis_title="Loss Components",
        yaxis_title="Percentage (%)",
        height=500,
        xaxis_tickangle=45
    )

    st.plotly_chart(bar_fig, use_container_width=True)



