# Final Complete app.py with Tabs and ML Features

import streamlit as st
import pandas as pd
import pyodbc
import io # For downloading plots/files
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ydata_profiling import ProfileReport # Updated name for pandas-profiling
import sweetviz as sv
import streamlit.components.v1 as components # For embedding HTML reports
import os # For path joining
import json # For loading DB config
import traceback # For more detailed error logging if needed

# --- ML Imports ---
from sklearn.model_selection import train_test_split # Though not used for training, good for context
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression, f_classif, chi2
from scipy.stats import skew

# --- Constants ---
DB_CONFIG_FILE = "db_config.json"
DEFAULT_ODBC_DRIVER = "{ODBC Driver 17 for SQL Server}" # Change if needed
DEFAULT_TRUSTED_CONNECTION = "yes" # Change to "no" if using UID/PWD by default
# If DEFAULT_TRUSTED_CONNECTION is "no", provide credentials below or use st.secrets
# DEFAULT_UID = st.secrets.db_credentials.username # Example using secrets
# DEFAULT_PWD = st.secrets.db_credentials.password # Example using secrets
DEFAULT_UID = "YOUR_USERNAME" # Fallback or direct use (less secure)
DEFAULT_PWD = "YOUR_PASSWORD" # Fallback or direct use (less secure)

# --- Load DB Configuration ---
def load_db_config(config_file):
    """Loads DB server names and addresses from a JSON file."""
    db_servers = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                db_servers = json.load(f)
        except json.JSONDecodeError:
            # Use st.error only if Streamlit context is available, might run early
            print(f"ERROR: Could not decode '{config_file}'. Please ensure it's valid JSON.")
            st.error(f"Error: Could not decode '{config_file}'. Please ensure it's valid JSON.")
        except Exception as e:
            print(f"ERROR: Error reading DB config file '{config_file}': {e}")
            st.error(f"Error reading DB config file '{config_file}': {e}")
    else:
        # Use st.warning only if Streamlit context is available
        print(f"WARNING: '{config_file}' not found. Database connection options will be limited.")
        # Delay the warning until app runs if needed:
        # st.session_state['db_config_warning'] = f"'{config_file}' not found. DB connections unavailable."
    return db_servers

DB_SERVERS = load_db_config(DB_CONFIG_FILE)
# Optionally display the warning during app execution
# if 'db_config_warning' in st.session_state:
#     st.sidebar.warning(st.session_state.db_config_warning)


# --- Action Definitions (for Wizard Tab) ---
ACTION_CATEGORIES = {
    "Basic Info": [
        "Show Shape", "Show Columns & Types", "Show Basic Statistics (Describe)",
        "Show Missing Values", "Show Unique Values (Categorical)", "Count Values (Categorical)"
    ],
    "Data Cleaning": [
        "Drop Columns", "Rename Columns", "Handle Missing Data",
        "Drop Duplicate Rows", "Change Data Type",
        "String Manipulation (Trim, Case, Replace)", "Extract from Text (Regex)",
        "Date Component Extraction (Year, Month...)"
    ],
    "Data Transformation": [
        "Filter Data", "Sort Data", "Select Columns", "Create Calculated Column (Basic Arithmetic)",
        "Bin Numeric Data (Cut)", "One-Hot Encode Categorical Column", "Pivot (Simple)" , "Melt (Unpivot)"
    ],
    "Aggregation & Analysis": [
        "Calculate Single Aggregation (Sum, Mean...)", "Group By & Aggregate",
        "Calculate Rolling Window Statistics", "Calculate Cumulative Statistics", "Rank Data"
    ],
    "Visualization": [
        "Plot Histogram (Numeric)", "Plot Density Plot (Numeric)", "Plot Count Plot (Categorical)",
        "Plot Bar Chart (Aggregated)", "Plot Line Chart", "Plot Scatter Plot",
        "Plot Box Plot (Numeric vs Cat)", "Plot Correlation Heatmap (Numeric)"
    ]
}

# --- Helper Functions (Database Connection Updated) ---

def build_connection_string(server_name, db_name=None):
    """Builds the connection string using loaded config and defaults."""
    if not server_name or server_name not in DB_SERVERS:
        st.error(f"Server '{server_name}' not found in configuration file '{DB_CONFIG_FILE}'.")
        return None

    server_address = DB_SERVERS[server_name]
    conn_str = f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};"
    if db_name:
        conn_str += f"DATABASE={db_name};"

    trusted_connection = DEFAULT_TRUSTED_CONNECTION.lower()
    conn_str += f"Trusted_Connection={trusted_connection};"

    if trusted_connection == 'no':
        # Attempt to get UID/PWD from defaults or potentially st.secrets
        # Replace with actual logic using st.secrets if implemented
        uid = DEFAULT_UID
        pwd = DEFAULT_PWD
        if uid and pwd:
             conn_str += f"UID={uid};PWD={pwd};"
        else:
            st.warning("Trusted Connection is 'no' but UID/PWD are not configured. Connection might fail.")
            # Optionally prevent connection attempt here
            # return None
    return conn_str

@st.cache_data(show_spinner="Connecting to database to fetch names...")
def get_databases(server_name):
    if not server_name: return []
    conn_str = build_connection_string(server_name) # Connect without specific DB first
    if not conn_str: return []

    databases = []
    try:
        # Increase timeout slightly for listing DBs
        with pyodbc.connect(conn_str, timeout=10, attrs_before={pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');")
            databases = [row.name for row in cursor.fetchall()]
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        st.error(f"Error connecting to {server_name} to list databases (SQLSTATE: {sqlstate}): {ex}. Check config/network/permissions.")
        if 'login failed' in str(ex).lower():
             st.error("Login failed. If using UID/PWD, check credentials. If Trusted Connection, check server permissions for the application user.")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching databases: {e}")
    return databases

@st.cache_data(show_spinner="Fetching table names...")
def get_tables(server_name, db_name):
    if not server_name or not db_name: return []
    conn_str = build_connection_string(server_name, db_name)
    if not conn_str: return []

    tables = []
    try:
        with pyodbc.connect(conn_str, timeout=10, attrs_before={pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME")
            tables = [f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}" for row in cursor.fetchall()]
    except pyodbc.Error as ex:
        st.error(f"Error connecting to {db_name} on {server_name} to list tables: {ex}. Check config/permissions.")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching tables: {e}")
    return tables

@st.cache_data(show_spinner="Fetching data from database...")
def fetch_data_from_db(server_name, db_name, _sql_query):
    df = pd.DataFrame()
    conn_str = build_connection_string(server_name, db_name)
    if not conn_str: return df # Return empty df if connection string failed

    try:
        # Increase timeout for potentially long queries
        with pyodbc.connect(conn_str, timeout=60, attrs_before={pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 60}) as conn:
            df = pd.read_sql(_sql_query, conn)
    except pyodbc.Error as ex:
        st.error(f"Database Error: {ex}. Failed to execute query. Check syntax and permissions.")
        st.code(_sql_query, language='sql') # Show the failed query
    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching: {e}")
    return df

# --- Other Helper Functions (Mostly Unchanged) ---

@st.cache_data
def load_file_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
    df = None
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            try:
                 excel_file = pd.ExcelFile(uploaded_file)
                 if len(excel_file.sheet_names) > 1:
                     st.info(f"Multiple sheets found in '{file_name}'. Loading the first sheet: '{excel_file.sheet_names[0]}'.")
                 df = excel_file.parse(excel_file.sheet_names[0])
            except Exception as excel_err:
                 st.error(f"Error reading Excel file '{file_name}': {excel_err}")
                 df = None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls, .xlsx).")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        st.error(traceback.format_exc())
        df = None
    return df

def get_column_types(df, include_text_guess=False):
    """Helper to categorize columns, optionally guessing 'text'."""
    if df is None:
        return [], [], [], [] if include_text_guess else [], []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64', 'timedelta']).columns.tolist()
    text_cols = [] # Initialize

    # Refine categorical: exclude high cardinality & optionally identify text
    potential_cat_cols = categorical_cols.copy()
    cat_threshold = 0.5 # Example: If > 50% unique values, maybe not categorical for plotting
    text_threshold_unique = 0.8 # If very high unique ratio in 'object' column
    text_threshold_avg_len = 50 # If average string length is high

    rows = len(df)
    if rows > 0:
        for col in potential_cat_cols:
            try:
                nunique = df[col].nunique(dropna=False)
                unique_ratio = nunique / rows

                # High cardinality check
                if unique_ratio > cat_threshold and nunique > 50: # Add absolute threshold too
                     if col in categorical_cols:
                         categorical_cols.remove(col)
                         # Check if it might be text
                         if include_text_guess:
                              is_potential_text = False
                              # Heuristic 1: Very high unique ratio
                              if unique_ratio > text_threshold_unique:
                                   is_potential_text = True
                              # Heuristic 2: Long average string length (if sample is string)
                              elif df[col].dtype == 'object':
                                  try:
                                      # Sample to avoid crashing on huge datasets/mixed types
                                      sample_avg_len = df[col].dropna().astype(str).str.len().head(1000).mean()
                                      if sample_avg_len > text_threshold_avg_len:
                                          is_potential_text = True
                                  except Exception: pass # Ignore errors during text heuristics

                              if is_potential_text:
                                  text_cols.append(col)

            except TypeError: # Handle complex types
                if col in categorical_cols:
                    categorical_cols.remove(col)
            except Exception: # Catch other potential errors during analysis
                 if col in categorical_cols:
                      categorical_cols.remove(col) # Be conservative

    if include_text_guess:
        # Ensure no overlap
        text_cols = list(set(text_cols) - set(numeric_cols) - set(datetime_cols))
        categorical_cols = list(set(categorical_cols) - set(text_cols))
        return numeric_cols, categorical_cols, datetime_cols, text_cols
    else:
         return numeric_cols, categorical_cols, datetime_cols


def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def save_plot_to_bytes(fig):
    img_bytes = io.BytesIO()
    try:
        fig.savefig(img_bytes, format='png', bbox_inches='tight', dpi=150)
        # Don't close figure here if st.pyplot is used afterwards in the same block
        # plt.close(fig) # Move closing after st.pyplot if needed
    except Exception as e:
         st.error(f"Error saving plot: {e}")
         # plt.close(fig) # Ensure figure is closed even on error if applicable
    img_bytes.seek(0)
    return img_bytes.getvalue()

# --- Caching functions for EDA Reports (Unchanged) ---
@st.cache_data(show_spinner="Generating YData Profile Report (this can take a while)...")
def generate_profile_report(_df, _title="Data Profile Report"):
    if _df is None or _df.empty: return None
    try:
        profile = ProfileReport(_df, title=_title, explorative=True, interactions=None,
                                correlations={"pearson": {"calculate": False}, "spearman": {"calculate": False},
                                              "kendall": {"calculate": False}, "phi_k": {"calculate": False},
                                              "cramers": {"calculate": False}},
                                missing_diagrams={"heatmap": False, "dendrogram": False} )
        return profile.to_html()
    except Exception as e:
        st.error(f"Error generating YData profile report: {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_data(show_spinner="Generating Sweetviz Report (this can take a while)...")
def generate_sweetviz_report(_df):
     if _df is None or _df.empty: return None
     try:
         report = sv.analyze(_df)
         html_report = report.show_html(filepath=None, open_browser=False, layout='vertical', scale=None)
         return html_report
     except NotImplementedError as nie:
         st.error(f"Sweetviz feature not implemented or failed: {nie}.")
         return None
     except Exception as e:
         st.error(f"Error generating Sweetviz report: {e}")
         st.error(traceback.format_exc())
         return None

# --- Initialize Session State ---
st.session_state.setdefault('data_source', None)
st.session_state.setdefault('df', None)
st.session_state.setdefault('db_params', {'server': None, 'database': None, 'table': None})
st.session_state.setdefault('generated_code', "") # Wizard: Last generated code
st.session_state.setdefault('result_type', None) # Wizard: Type of result
st.session_state.setdefault('uploaded_file_state', None)
st.session_state.setdefault('selected_category', list(ACTION_CATEGORIES.keys())[0]) # Wizard category
st.session_state.setdefault('show_profile_report', False) # Wizard tab flag
st.session_state.setdefault('show_sweetviz_report', False) # Wizard tab flag
st.session_state.setdefault('current_action_result_display', None) # Wizard: Result to display

# --- App Layout ---
st.set_page_config(layout="wide", page_title="AdViz+ ML Insights")
st.title("📊 AdViz+ ML Insights")
st.markdown("Explore your data, prepare it for ML, and gain insights.")

# --- Sidebar for Data Connection & Session Control (Mostly Unchanged) ---
with st.sidebar:
    st.header("🔄 Session Control")
    if st.button("Start New Session / Reset All"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear: del st.session_state[key]
        st.rerun()

    st.divider()
    st.header("🔗 Connect to Data")
    data_source_option = st.radio(
        "Choose your data source:",
        ('Upload File (CSV/Excel)', 'Connect to Database'),
        key='data_source_radio',
        index=0 if st.session_state.data_source == 'Upload File (CSV/Excel)' else 1 if st.session_state.data_source == 'Connect to Database' else None,
        on_change=lambda: st.session_state.update(df=None, generated_code="", result_type=None, current_action_result_display=None, show_profile_report=False, show_sweetviz_report=False)
    )
    if data_source_option != st.session_state.data_source:
         st.session_state.data_source = data_source_option
         st.rerun()

    # --- File Upload Section ---
    if st.session_state.data_source == 'Upload File (CSV/Excel)':
        st.subheader("📤 Upload File")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'], key='file_uploader')
        if uploaded_file is not None:
            file_identifier = (uploaded_file.name, uploaded_file.size, uploaded_file.type)
            if st.session_state.uploaded_file_state != file_identifier:
                st.session_state.uploaded_file_state = file_identifier
                st.session_state.df = None
                st.session_state.generated_code = ""
                st.session_state.result_type = None
                st.session_state.show_profile_report = False
                st.session_state.show_sweetviz_report = False
                st.session_state.current_action_result_display = None
                st.session_state.df = load_file_data(uploaded_file)
                if st.session_state.df is not None:
                    st.success(f"Successfully loaded `{uploaded_file.name}`")
                    st.rerun()
                else:
                    st.session_state.uploaded_file_state = None

    # --- Database Connection Section (Updated to use new config) ---
    elif st.session_state.data_source == 'Connect to Database':
        st.subheader("🗄️ Connect to Database")
        if not DB_SERVERS:
             st.warning(f"Database server configuration ('{DB_CONFIG_FILE}') is missing or empty. Cannot connect.")
        else:
            available_servers = list(DB_SERVERS.keys())
            selected_server = st.selectbox(
                "1. Select Server", options=available_servers, index=available_servers.index(st.session_state.db_params['server']) if st.session_state.db_params['server'] in available_servers else None, placeholder="Choose a server...", key="db_server_select"
            )

            if selected_server != st.session_state.db_params['server']:
                 st.session_state.db_params['server'] = selected_server
                 st.session_state.db_params['database'] = None
                 st.session_state.db_params['table'] = None
                 st.session_state.df = None
                 st.rerun()

            selected_db = None
            if selected_server:
                available_dbs = get_databases(selected_server) # Uses new function
                if available_dbs:
                    selected_db = st.selectbox(
                        "2. Select Database", options=available_dbs, index=available_dbs.index(st.session_state.db_params['database']) if st.session_state.db_params['database'] in available_dbs else None, placeholder="Choose a database...", key="db_select"
                    )
                    if selected_db != st.session_state.db_params['database']:
                        st.session_state.db_params['database'] = selected_db
                        st.session_state.db_params['table'] = None
                        st.session_state.df = None
                        st.rerun()
                else:
                    st.session_state.db_params['database'] = None

            selected_table = None
            if selected_server and st.session_state.db_params['database']:
                current_db = st.session_state.db_params['database']
                if current_db:
                    available_tables = get_tables(selected_server, current_db) # Uses new function
                    if available_tables:
                        selected_table = st.selectbox(
                            "3. Select Table", options=available_tables, index=available_tables.index(st.session_state.db_params['table']) if st.session_state.db_params['table'] in available_tables else None, placeholder="Choose a table...", key="db_table_select"
                        )
                        if selected_table != st.session_state.db_params['table']:
                             st.session_state.db_params['table'] = selected_table
                             st.session_state.df = None
                             st.rerun()
                    else:
                        st.session_state.db_params['table'] = None

            current_table = st.session_state.db_params['table']
            if current_table:
                query_method = st.radio("4. Fetch Method", ("Select TOP 1000 Rows", "Custom SQL Query"), key="db_query_method")
                sql_query = ""
                if query_method == "Select TOP 1000 Rows":
                     table_parts = current_table.split('.')
                     quoted_table = f"[{table_parts[0]}].[{table_parts[1]}]" if len(table_parts) == 2 else f"[{current_table}]"
                     sql_query = f"SELECT TOP 1000 * FROM {quoted_table};"
                     st.text_area("Generated SQL:", value=sql_query, height=100, disabled=True, key="db_sql_display")
                else:
                    default_custom_sql = f"SELECT * FROM {current_table} WHERE ..." if current_table else "SELECT column1, column2 FROM your_table WHERE condition;"
                    sql_query = st.text_area("Enter your SQL Query:", value=st.session_state.get('custom_sql_input', default_custom_sql), height=150, key="db_sql_custom", on_change=lambda: st.session_state.update(custom_sql_input=st.session_state.db_sql_custom))

                if st.button("Fetch Data from Database", key="db_fetch_button"):
                    current_sql_to_run = sql_query
                    current_db_name = st.session_state.db_params['database']
                    if current_sql_to_run and selected_server and current_db_name:
                        # Reset state before fetching
                        st.session_state.df = None
                        st.session_state.generated_code = ""
                        st.session_state.result_type = None
                        st.session_state.show_profile_report = False
                        st.session_state.show_sweetviz_report = False
                        st.session_state.current_action_result_display = None
                        st.session_state.uploaded_file_state = None

                        # Fetch data using updated function
                        fetched_df = fetch_data_from_db(selected_server, current_db_name, current_sql_to_run)
                        st.session_state.df = fetched_df

                        if fetched_df is not None and not fetched_df.empty:
                            st.success(f"Successfully fetched data using query.")
                            st.rerun()
                        elif fetched_df is not None and fetched_df.empty:
                            st.warning("Query executed successfully but returned no data.")
                            st.rerun()
                        # Errors handled within fetch_data_from_db
                    else:
                        st.warning("Please ensure server, database, and table/query are correctly configured.")


# --- Main Application Area with Tabs ---
if st.session_state.df is not None:
    df = st.session_state.df # Convenience variable
    all_cols = df.columns.tolist()
    # Get types once for use in multiple tabs
    numeric_cols, categorical_cols, datetime_cols, text_cols = get_column_types(df, include_text_guess=True) # Guess text columns

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Analysis Wizard",
        "🤖 ML Readiness Assessment",
        "🛠️ ML Preprocessing Simulation",
        "💡 XAI Feature Insights",
        "🧠 Model Selection Advisor"
    ])

    # =============================================
    # == Tab 1: Original Data Analysis Wizard =====
    # =============================================
    with tab1:
        st.header("Preview of Loaded Data")
        st.dataframe(df, height=300, use_container_width=True)

        # --- Automated EDA Report Buttons ---
        st.subheader("📊 Automated EDA Reports (Add-on)")
        st.markdown("Generate comprehensive reports for quick data understanding.")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            if st.button("Generate YData Profile Report", key="profile_report_btn_tab1", use_container_width=True):
                st.session_state.show_profile_report = True
                st.session_state.show_sweetviz_report = False
                st.session_state.current_action_result_display = None
                st.rerun()
        with report_col2:
            if st.button("Generate Sweetviz Report", key="sweetviz_report_btn_tab1", use_container_width=True):
                st.session_state.show_sweetviz_report = True
                st.session_state.show_profile_report = False
                st.session_state.current_action_result_display = None
                st.rerun()

        # --- Display EDA Reports Conditionally ---
        report_placeholder_tab1 = st.empty()
        with report_placeholder_tab1.container():
            if st.session_state.get('show_profile_report', False):
                report_title = f"Profile Report for {st.session_state.uploaded_file_state[0]}" if st.session_state.uploaded_file_state else "Profile Report"
                report_html = generate_profile_report(df, report_title)
                if report_html:
                    with st.expander("YData Profile Report", expanded=True):
                        components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate YData Profile Report.")

            if st.session_state.get('show_sweetviz_report', False):
                report_html = generate_sweetviz_report(df)
                if report_html:
                    with st.expander("Sweetviz Report", expanded=True):
                        components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate Sweetviz Report.")

        st.divider()

        # --- Data Analysis Wizard ---
        st.header("🧙‍♂️ Data Analysis Wizard")
        st.markdown("Perform common data tasks without coding.")

        # Re-fetch types specifically for wizard if needed (might differ slightly from ML heuristic)
        wiz_numeric_cols, wiz_categorical_cols, wiz_datetime_cols = get_column_types(df)

        st.subheader("1. Select Action Category")
        selected_category_index = list(ACTION_CATEGORIES.keys()).index(st.session_state.selected_category)
        selected_category = st.radio(
            "Category:", list(ACTION_CATEGORIES.keys()), index=selected_category_index, horizontal=True, key="category_radio_tab1"
        )
        if selected_category != st.session_state.selected_category:
            st.session_state.selected_category = selected_category
            st.rerun()

        st.subheader("2. Choose Specific Action & Configure")
        action_options = ACTION_CATEGORIES[selected_category]
        st.session_state.setdefault(f'selected_action_{selected_category}', action_options[0])
        selected_action_index = action_options.index(st.session_state[f'selected_action_{selected_category}']) if st.session_state[f'selected_action_{selected_category}'] in action_options else 0
        action = st.selectbox(
            f"Select Action in '{selected_category}':", action_options, index=selected_action_index, key=f"action_select_{selected_category}_tab1"
        )
        st.session_state[f'selected_action_{selected_category}'] = action

        # --- Wizard Action Implementation Block (Copied & Pasted - KEPT LONG FOR COMPLETENESS) ---
        # (This is the large block from the original script, slightly adapted keys)
        code_string = "# Select an action and configure options"
        result_data_config = None
        fig_config = None
        action_configured = False

        try:
            # === Basic Info ===
            if action == "Show Shape":
                code_string = "result_data = df.shape"
                action_configured = True
            elif action == "Show Columns & Types":
                code_string = "result_data = df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'DataType'})"
                action_configured = True
            elif action == "Show Basic Statistics (Describe)":
                sel_cols_describe = st.multiselect("Select Numeric Columns (Optional, default=all numeric):", wiz_numeric_cols, default=wiz_numeric_cols, key=f"desc_{action}_t1")
                code_string = f"result_data = df[{sel_cols_describe}].describe()" if sel_cols_describe else "result_data = df.describe(include='all')"
                action_configured = True
            elif action == "Show Missing Values":
                code_string = "result_data = df.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'}).query('`Missing Count` > 0')"
                action_configured = True
            elif action == "Show Unique Values (Categorical)":
                cat_col_unique = st.selectbox("Select Categorical Column:", wiz_categorical_cols, key=f"unique_{action}_t1")
                if cat_col_unique:
                    code_string = f"result_data = pd.DataFrame(df['{cat_col_unique}'].unique(), columns=['Unique Values'])"
                    action_configured = True
            elif action == "Count Values (Categorical)":
                 cat_col_count = st.selectbox("Select Categorical Column:", wiz_categorical_cols, key=f"count_{action}_t1")
                 if cat_col_count:
                    normalize = st.checkbox("Show as Percentage (%)", key=f"count_norm_{action}_t1")
                    code_string = f"counts = df['{cat_col_count}'].value_counts(normalize={normalize}).reset_index()\ncounts.columns = ['{cat_col_count}', 'Percentage' if {normalize} else 'Count']\nresult_data = counts"
                    action_configured = True

            # === Data Cleaning ===
            elif action == "Drop Columns":
                 drop_cols = st.multiselect("Select Columns to Drop:", all_cols, key=f"drop_{action}_t1")
                 if drop_cols:
                     code_string = f"result_data = df.drop(columns={drop_cols})"
                     action_configured = True
            elif action == "Rename Columns":
                rename_map = {}
                st.write("Select columns and enter new names:")
                cols_to_rename = st.multiselect("Columns to Rename:", all_cols, key=f"rename_select_{action}_t1")
                for col in cols_to_rename:
                    new_name = st.text_input(f"New name for '{col}':", value=col, key=f"rename_input_{col}_{action}_t1")
                    if new_name != col and new_name:
                        rename_map[col] = new_name
                if rename_map:
                    code_string = f"result_data = df.rename(columns={rename_map})"
                    action_configured = True
                elif cols_to_rename: st.caption("Enter new names for the selected columns.")

            elif action == "Handle Missing Data":
                 fill_cols = st.multiselect("Select Columns to Fill NA:", all_cols, key=f"fillna_cols_{action}_t1")
                 if fill_cols:
                     fill_method = st.radio("Fill Method:", ["Specific Value", "Mean", "Median", "Mode", "Forward Fill (ffill)", "Backward Fill (bfill)", "Drop Rows with NA"], key=f"fillna_method_{action}_t1")
                     code_lines = ["result_data = df.copy()"]
                     valid_op = True
                     if fill_method == "Specific Value":
                         fill_value = st.text_input("Enter Value to Fill NA with:", "0", key=f"fillna_value_{action}_t1")
                         try: fill_value_parsed = float(fill_value)
                         except ValueError: fill_value_parsed = fill_value
                         for col in fill_cols: code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna({repr(fill_value_parsed)})")
                     elif fill_method == "Drop Rows with NA": code_lines.append(f"result_data = result_data.dropna(subset={fill_cols})")
                     else: # Mean, Median, Mode, ffill, bfill
                         for col in fill_cols:
                             if fill_method in ["Mean", "Median"] and col not in wiz_numeric_cols:
                                  st.warning(f"Cannot apply '{fill_method}' to non-numeric column '{col}'. Skipping.")
                                  valid_op = False; break
                             elif fill_method == "Mean": code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna(result_data['{col}'].mean())")
                             elif fill_method == "Median": code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna(result_data['{col}'].median())")
                             elif fill_method == "Mode": code_lines.append(f"if not result_data['{col}'].mode().empty: result_data['{col}'] = result_data['{col}'].fillna(result_data['{col}'].mode()[0])")
                             elif fill_method == "Forward Fill (ffill)": code_lines.append(f"result_data['{col}'] = result_data['{col}'].ffill()")
                             elif fill_method == "Backward Fill (bfill)": code_lines.append(f"result_data['{col}'] = result_data['{col}'].bfill()")
                     if valid_op:
                        code_string = "\n".join(code_lines)
                        action_configured = True

            elif action == "Drop Duplicate Rows":
                subset_cols = st.multiselect("Consider Columns (Optional, default=all):", all_cols, key=f"dropdup_subset_{action}_t1")
                keep_option = st.radio("Keep Which Duplicate:", ['first', 'last', False], format_func=lambda x: str(x) if isinstance(x,str) else "Drop All Duplicates", key=f"dropdup_keep_{action}_t1")
                code_string = f"result_data = df.drop_duplicates(subset={subset_cols if subset_cols else None}, keep='{keep_option}' if {repr(keep_option)} != 'False' else False)"
                action_configured = True

            elif action == "Change Data Type":
                type_col = st.selectbox("Select Column:", all_cols, key=f"dtype_col_{action}_t1")
                target_type = st.selectbox("Convert To Type:", ['int', 'float', 'str', 'datetime', 'category'], key=f"dtype_target_{action}_t1")
                if type_col and target_type:
                    code_lines = ["result_data = df.copy()"]
                    if target_type == 'datetime':
                         infer_format = st.checkbox("Try to Infer Datetime Format", value=True, key=f"dtype_infer_{action}_t1")
                         format_str = f", infer_datetime_format={infer_format}" if infer_format else ""
                         code_lines.append(f"result_data['{type_col}'] = pd.to_datetime(result_data['{type_col}'], errors='coerce'{format_str})")
                    elif target_type in ['int', 'float']:
                         code_lines.append(f"result_data['{type_col}'] = pd.to_numeric(result_data['{type_col}'], errors='coerce')")
                         if target_type == 'int': code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].astype('Int64')") # Use nullable Int64
                    else: code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].astype('{target_type}')")
                    code_string = "\n".join(code_lines)
                    action_configured = True

            elif action == "String Manipulation (Trim, Case, Replace)":
                str_col = st.selectbox("Select Column:", all_cols, key=f"strman_col_{action}_t1")
                if str_col:
                    str_op = st.radio("Operation:", ["Trim Whitespace", "To Uppercase", "To Lowercase", "To Title Case", "Replace Text"], key=f"strman_op_{action}_t1")
                    code_lines = ["result_data = df.copy()"]
                    base_str_col = f"result_data['{str_col}'].astype(str)"
                    if str_op == "Trim Whitespace": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.strip()")
                    elif str_op == "To Uppercase": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.upper()")
                    elif str_op == "To Lowercase": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.lower()")
                    elif str_op == "To Title Case": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.title()")
                    elif str_op == "Replace Text":
                        text_to_find = st.text_input("Text to Find:", key=f"strman_find_{action}_t1")
                        text_to_replace = st.text_input("Replace With:", key=f"strman_replace_{action}_t1")
                        use_regex = st.checkbox("Use Regular Expression", key=f"strman_regex_{action}_t1")
                        if text_to_find:
                            code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.replace({repr(text_to_find)}, {repr(text_to_replace)}, regex={use_regex})")
                            action_configured = True
                        else: st.caption("Enter text to find."); action_configured = False
                    else: action_configured = True
                    if action_configured: code_string = "\n".join(code_lines)

            elif action == "Extract from Text (Regex)":
                extract_col = st.selectbox("Select Column to Extract From:", all_cols, key=f"extract_col_{action}_t1")
                regex_pattern = st.text_input("Enter Regex Pattern (with capture groups):", placeholder=r"(\d+)-(\w+)", key=f"extract_regex_{action}_t1")
                new_col_names_str = st.text_input("New Column Names (comma-separated, optional):", placeholder="NumPart,TextPart", key=f"extract_names_{action}_t1")
                if extract_col and regex_pattern:
                    code_lines = ["result_data = df.copy()"]
                    new_col_names = [name.strip() for name in new_col_names_str.split(',') if name.strip()] if new_col_names_str else None
                    new_col_assign = f"result_data[{repr(new_col_names)}]" if new_col_names else "extracted_data"
                    code_lines.append(f"{new_col_assign} = result_data['{extract_col}'].astype(str).str.extract(r{repr(regex_pattern)})")
                    if not new_col_names: code_lines.append("result_data = extracted_data")
                    code_string = "\n".join(code_lines)
                    action_configured = True

            elif action == "Date Component Extraction (Year, Month...)":
                 date_col = st.selectbox("Select Column (will attempt Datetime conversion):", all_cols, key=f"datecomp_col_{action}_t1")
                 component = st.selectbox("Component to Extract:", ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Day Name", "Month Name", "Quarter", "Week of Year"], key=f"datecomp_comp_{action}_t1")
                 if date_col and component:
                     comp_map = {"Year": ".dt.year", "Month": ".dt.month", "Day": ".dt.day", "Hour": ".dt.hour", "Minute": ".dt.minute", "Second": ".dt.second", "Day of Week": ".dt.dayofweek", "Day Name": ".dt.day_name()", "Month Name": ".dt.month_name()", "Quarter": ".dt.quarter", "Week of Year": ".dt.isocalendar().week"}
                     new_date_col_name = f"{date_col}_{component.lower().replace(' ', '_')}"
                     code_lines = ["result_data = df.copy()", f"temp_date_col = pd.to_datetime(result_data['{date_col}'], errors='coerce')", f"result_data['{new_date_col_name}'] = temp_date_col{comp_map[component]}"]
                     code_string = "\n".join(code_lines)
                     action_configured = True

            # === Data Transformation ===
            elif action == "Filter Data":
                filter_col = st.selectbox("Select Column to Filter:", all_cols, key=f"filter_col_{action}_t1")
                if filter_col:
                    col_is_numeric = pd.api.types.is_numeric_dtype(df[filter_col])
                    col_is_datetime = pd.api.types.is_datetime64_any_dtype(df[filter_col]) or 'date' in filter_col.lower()
                    operators_num = ['==', '!=', '>', '<', '>=', '<=']
                    operators_str = ['==', '!=', 'contains', 'starts with', 'ends with', 'is in (comma-sep list)', 'is not in (comma-sep list)']
                    operators_date = ['==', '!=', '>', '<', '>=', '<=']
                    val = None
                    if col_is_numeric:
                        op = st.selectbox("Operator:", operators_num, key=f"filter_op_num_{action}_t1")
                        val = st.number_input(f"Value:", value=0.0, format="%g", key=f"filter_val_num_{filter_col}_t1")
                        code_string = f"result_data = df[pd.to_numeric(df['{filter_col}'], errors='coerce') {op} {val}]" # Added to_numeric
                        action_configured = True
                    elif col_is_datetime:
                         op = st.selectbox("Operator:", operators_date, key=f"filter_op_date_{action}_t1")
                         val_date = st.date_input(f"Date:", key=f"filter_val_date_{filter_col}_t1")
                         val = pd.Timestamp(f"{val_date}")
                         code_string = f"temp_date_col = pd.to_datetime(df['{filter_col}'], errors='coerce')\nresult_data = df[temp_date_col {op} pd.Timestamp('{val}')]"
                         action_configured = True
                    else: # String/Categorical
                        op = st.selectbox("Operator:", operators_str, key=f"filter_op_str_{action}_t1")
                        val_str = st.text_input(f"Value(s):", key=f"filter_val_str_{filter_col}_t1")
                        if val_str or op in ['is in (comma-sep list)', 'is not in (comma-sep list)']: # Allow empty list check? Maybe better not to.
                            filter_col_as_str = f"df['{filter_col}'].astype(str)"
                            if op == 'contains': code_string = f"result_data = df[{filter_col_as_str}.str.contains({repr(val_str)}, case=False, na=False)]"
                            elif op == 'starts with': code_string = f"result_data = df[{filter_col_as_str}.str.startswith({repr(val_str)}, na=False)]"
                            elif op == 'ends with': code_string = f"result_data = df[{filter_col_as_str}.str.endswith({repr(val_str)}, na=False)]"
                            elif op == 'is in (comma-sep list)':
                                list_vals = [repr(v.strip()) for v in val_str.split(',') if v.strip()]
                                if list_vals: code_string = f"result_data = df[df['{filter_col}'].astype(str).isin([{', '.join(list_vals)}])]"
                                else: st.caption("Enter comma-separated values."); action_configured = False
                            elif op == 'is not in (comma-sep list)':
                                list_vals = [repr(v.strip()) for v in val_str.split(',') if v.strip()]
                                code_string = f"result_data = df[~df['{filter_col}'].astype(str).isin([{', '.join(list_vals)}])]"
                            else: # == or !=
                                code_string = f"result_data = df[df['{filter_col}'].astype(str) {op} {repr(val_str)}]"
                            action_configured = bool(code_string) # Set configured if code was generated


            elif action == "Sort Data":
                sort_cols = st.multiselect("Sort By Column(s):", all_cols, key=f"sort_cols_{action}_t1")
                if sort_cols:
                    sort_orders_bool = []
                    for col in sort_cols:
                        order = st.radio(f"Order for '{col}':", ["Ascending", "Descending"], key=f"sort_{col}_{action}_t1", horizontal=True)
                        sort_orders_bool.append(order == "Ascending")
                    code_string = f"result_data = df.sort_values(by={sort_cols}, ascending={sort_orders_bool})"
                    action_configured = True

            elif action == "Select Columns":
                 select_cols = st.multiselect("Select Columns to Keep:", all_cols, default=all_cols, key=f"select_cols_{action}_t1")
                 if select_cols:
                     code_string = f"result_data = df[{select_cols}]"
                     action_configured = True
                 else: st.warning("Select at least one column.")

            elif action == "Create Calculated Column (Basic Arithmetic)":
                 st.write("Create column using +, -, *, / on two numeric columns or column and constant.")
                 new_calc_col_name = st.text_input("New Column Name:", key=f"calc_newname_{action}_t1")
                 col1 = st.selectbox("First Numeric Column (or None):", [None] + wiz_numeric_cols, key=f"calc_col1_{action}_t1")
                 op_calc = st.selectbox("Operator:", ['+', '-', '*', '/'], key=f"calc_op_{action}_t1")
                 col2 = st.selectbox("Second Numeric Column (or None):", [None] + wiz_numeric_cols, key=f"calc_col2_{action}_t1")
                 constant_val_str = st.text_input("Or Constant Value:", "0", key=f"calc_const_{action}_t1")
                 if new_calc_col_name and op_calc and (col1 or col2 or constant_val_str):
                     try:
                         constant_val = float(constant_val_str) if (not col1 or not col2) else 0.0
                         term1 = f"pd.to_numeric(df['{col1}'], errors='coerce')" if col1 else str(constant_val)
                         term2 = f"pd.to_numeric(df['{col2}'], errors='coerce')" if col2 else str(constant_val)
                         if col1 or col2:
                              if op_calc == '/' and not col2 and constant_val == 0: st.error("Cannot divide by zero constant.")
                              else:
                                 code_string = f"result_data = df.copy()\nresult_data['{new_calc_col_name}'] = {term1} {op_calc} {term2}"
                                 action_configured = True
                         else: st.warning("Select at least one column.")
                     except ValueError: st.error("Invalid constant value.")
                 elif not new_calc_col_name: st.caption("Enter a name for the new column.")

            elif action == "Bin Numeric Data (Cut)":
                bin_col = st.selectbox("Select Numeric Column to Bin:", wiz_numeric_cols, key=f"bin_col_{action}_t1")
                if bin_col:
                     bin_method = st.radio("Method:", ["Equal Width", "Quantile Based", "Custom Edges"], key=f"bin_method_{action}_t1")
                     new_bin_col_name = st.text_input("New Binned Column Name:", f"{bin_col}_binned", key=f"bin_newname_{action}_t1")
                     if not new_bin_col_name: new_bin_col_name = f"{bin_col}_binned"
                     code_lines = ["result_data = df.copy()", f"numeric_col = pd.to_numeric(result_data['{bin_col}'], errors='coerce')"]
                     bin_labels_type = st.radio("Label Type:", ["Numeric (0, 1..)", "Range Labels"], key=f"bin_labels_{action}_t1", horizontal=True)
                     labels_param = "labels=False" if bin_labels_type == "Numeric (0, 1..)" else ""
                     if bin_method == "Equal Width":
                         num_bins = st.slider("Number of Bins:", 2, 50, 5, key=f"bin_num_eq_{action}_t1")
                         code_lines.append(f"result_data['{new_bin_col_name}'] = pd.cut(numeric_col, bins={num_bins}, {labels_param}, include_lowest=True, duplicates='drop')")
                         action_configured = True
                     elif bin_method == "Quantile Based":
                         num_q_bins = st.slider("Number of Quantile Bins:", 2, 10, 4, key=f"bin_num_q_{action}_t1")
                         code_lines.append(f"result_data['{new_bin_col_name}'] = pd.qcut(numeric_col, q={num_q_bins}, {labels_param}, duplicates='drop')")
                         action_configured = True
                     elif bin_method == "Custom Edges":
                         edges_str = st.text_input("Enter Bin Edges (comma-sep):", "0, 10, 50, 100", key=f"bin_edges_{action}_t1")
                         try:
                             edges = sorted([float(e.strip()) for e in edges_str.split(',') if e.strip()])
                             if len(edges) > 1:
                                 code_lines.append(f"bin_edges = {edges}")
                                 code_lines.append(f"result_data['{new_bin_col_name}'] = pd.cut(numeric_col, bins=bin_edges, {labels_param}, include_lowest=True, duplicates='drop')")
                                 action_configured = True
                             elif edges_str: st.warning("Provide at least two valid edges.")
                         except ValueError: st.error("Invalid bin edges.")
                     if action_configured: code_string = "\n".join(code_lines)

            elif action == "One-Hot Encode Categorical Column":
                 ohe_col = st.selectbox("Select Categorical Column:", wiz_categorical_cols, key=f"ohe_col_{action}_t1")
                 if ohe_col:
                     drop_first = st.checkbox("Drop First Category", value=False, key=f"ohe_drop_{action}_t1")
                     code_string = f"result_data = pd.get_dummies(df, columns=['{ohe_col}'], prefix='{ohe_col}', drop_first={drop_first})"
                     action_configured = True

            elif action == "Pivot (Simple)":
                 st.info("Create pivot table summary.")
                 pivot_index = st.selectbox("Index (Rows):", all_cols, key=f"pivot_idx_{action}_t1")
                 pivot_cols = st.selectbox("Columns:", all_cols, key=f"pivot_cols_{action}_t1")
                 pivot_vals = st.selectbox("Values (Numeric):", wiz_numeric_cols, key=f"pivot_vals_{action}_t1")
                 pivot_agg = st.selectbox("Agg Function:", ['mean', 'sum', 'count', 'median', 'min', 'max'], key=f"pivot_agg_{action}_t1")
                 if pivot_index and pivot_cols and pivot_vals and pivot_agg:
                     if pivot_index == pivot_cols: st.warning("Index and Columns cannot be the same.")
                     else:
                          code_string = f"result_data = pd.pivot_table(df, index='{pivot_index}', columns='{pivot_cols}', values='{pivot_vals}', aggfunc='{pivot_agg}').reset_index()"
                          action_configured = True

            elif action == "Melt (Unpivot)":
                 st.info("Unpivot from wide to long format.")
                 id_vars = st.multiselect("Identifier Variables (Keep):", all_cols, key=f"melt_idvars_{action}_t1")
                 default_value_vars = [c for c in all_cols if c not in id_vars]
                 value_vars = st.multiselect("Value Variables (Unpivot, Opt.):", [c for c in all_cols if c not in id_vars], default=default_value_vars, key=f"melt_valvars_{action}_t1")
                 var_name = st.text_input("New Variable Column Name:", "Variable", key=f"melt_varname_{action}_t1")
                 value_name = st.text_input("New Value Column Name:", "Value", key=f"melt_valuename_{action}_t1")
                 if id_vars and var_name and value_name:
                     value_vars_param = f"value_vars={value_vars}" if value_vars else ""
                     code_string = f"result_data = pd.melt(df, id_vars={id_vars}, {value_vars_param}, var_name='{var_name}', value_name='{value_name}')"
                     action_configured = True
                 elif not id_vars: st.warning("Select at least one Identifier Variable.")

            # === Aggregation & Analysis ===
            elif action == "Calculate Single Aggregation (Sum, Mean...)":
                agg_col = st.selectbox("Select Numeric Column:", wiz_numeric_cols, key=f"sagg_col_{action}_t1")
                agg_func = st.selectbox("Function:", ['sum', 'mean', 'median', 'min', 'max', 'count', 'nunique', 'std', 'var'], key=f"sagg_func_{action}_t1")
                if agg_col and agg_func:
                    code_string = f"agg_numeric_col = pd.to_numeric(df['{agg_col}'], errors='coerce')\nresult_data = agg_numeric_col.{agg_func}()"
                    action_configured = True

            elif action == "Group By & Aggregate":
                group_cols = st.multiselect("Group By Column(s):", wiz_categorical_cols + wiz_datetime_cols, key=f"gagg_groupcols_{action}_t1")
                st.write("Define Aggregations:")
                num_aggs = st.number_input("Number of Aggregations:", min_value=1, value=1, key=f"gagg_numaggs_{action}_t1")
                named_aggs_list = []
                valid_agg_config = True
                for i in range(num_aggs):
                    st.markdown(f"**Agg #{i+1}**")
                    agg_col_group = st.selectbox(f"Aggregate Column:", wiz_numeric_cols, key=f"gagg_aggcol_{i}_{action}_t1")
                    agg_func_group = st.selectbox(f"Function:", ['sum', 'mean', 'median', 'min', 'max', 'count', 'nunique', 'std', 'var'], key=f"gagg_aggfunc_{i}_{action}_t1")
                    default_name = f"{agg_col_group}_{agg_func_group}".replace('[^A-Za-z0-9_]+', '') if agg_col_group else f"agg_{i+1}"
                    new_agg_name = st.text_input(f"Result Column Name:", value=default_name, key=f"gagg_aggname_{i}_{action}_t1")
                    if not agg_col_group: st.warning(f"Agg #{i+1}: Select column."); valid_agg_config = False
                    elif not new_agg_name or not new_agg_name.isidentifier(): st.warning(f"Agg #{i+1}: Enter valid name."); valid_agg_config = False
                    else: named_aggs_list.append(f"{new_agg_name}=pd.NamedAgg(column='{agg_col_group}', aggfunc='{agg_func_group}')")
                if group_cols and named_aggs_list and valid_agg_config:
                     named_aggs_str = ", ".join(named_aggs_list)
                     code_string = f"# Ensure numeric columns are numeric\nresult_data = df.groupby({group_cols}).agg({named_aggs_str}).reset_index()"
                     action_configured = True
                elif not group_cols: st.warning("Select at least one Group By column.")

            elif action == "Calculate Rolling Window Statistics":
                roll_col = st.selectbox("Select Numeric Column:", wiz_numeric_cols, key=f"roll_col_{action}_t1")
                if roll_col:
                    window_size = st.number_input("Window Size:", min_value=2, value=3, key=f"roll_window_{action}_t1")
                    roll_func = st.selectbox("Rolling Function:", ['mean', 'sum', 'median', 'std', 'min', 'max', 'var'], key=f"roll_func_{action}_t1")
                    center = st.checkbox("Center Window?", value=False, key=f"roll_center_{action}_t1")
                    min_periods = st.number_input("Min Periods (Opt.):", min_value=1, value=None, placeholder="Default", key=f"roll_minp_{action}_t1")
                    sort_by_col = st.selectbox("Sort data by (Optional):", [None] + all_cols, key=f"roll_sort_{action}_t1")
                    new_roll_col_name = st.text_input("New Column Name:", f"{roll_col}_roll_{roll_func}_{window_size}", key=f"roll_newname_{action}_t1")
                    if not new_roll_col_name: new_roll_col_name = f"{roll_col}_roll_{roll_func}_{window_size}"
                    code_lines = ["result_data = df.copy()"]
                    if sort_by_col: code_lines.append(f"result_data = result_data.sort_values(by='{sort_by_col}')")
                    code_lines.append(f"rolling_col_numeric = pd.to_numeric(result_data['{roll_col}'], errors='coerce')")
                    min_periods_param = f", min_periods={min_periods}" if min_periods is not None else ""
                    code_lines.append(f"result_data['{new_roll_col_name}'] = rolling_col_numeric.rolling(window={window_size}, center={center}{min_periods_param}).{roll_func}()")
                    # if sort_by_col: code_lines.append(f"result_data = result_data.sort_index()") # Optional: Restore order
                    code_string = "\n".join(code_lines)
                    action_configured = True

            elif action == "Calculate Cumulative Statistics":
                 cum_col = st.selectbox("Select Numeric Column:", wiz_numeric_cols, key=f"cum_col_{action}_t1")
                 if cum_col:
                     cum_func = st.selectbox("Cumulative Function:", ['sum', 'prod', 'min', 'max'], key=f"cum_func_{action}_t1")
                     sort_by_col_cum = st.selectbox("Sort data by (Optional):", [None] + all_cols, key=f"cum_sort_{action}_t1")
                     new_cum_col_name = st.text_input("New Column Name:", f"{cum_col}_cum_{cum_func}", key=f"cum_newname_{action}_t1")
                     if not new_cum_col_name: new_cum_col_name = f"{cum_col}_cum_{cum_func}"
                     code_lines = ["result_data = df.copy()"]
                     if sort_by_col_cum: code_lines.append(f"result_data = result_data.sort_values(by='{sort_by_col_cum}')")
                     code_lines.append(f"cum_col_numeric = pd.to_numeric(result_data['{cum_col}'], errors='coerce')")
                     code_lines.append(f"result_data['{new_cum_col_name}'] = cum_col_numeric.cum{cum_func}()")
                     # if sort_by_col_cum: code_lines.append(f"result_data = result_data.sort_index()")
                     code_string = "\n".join(code_lines)
                     action_configured = True

            elif action == "Rank Data":
                 rank_col = st.selectbox("Select Numeric Column to Rank By:", wiz_numeric_cols, key=f"rank_col_{action}_t1")
                 if rank_col:
                     rank_ascending = st.radio("Rank Order:", ["Ascending", "Descending"], key=f"rank_order_{action}_t1", horizontal=True) == "Ascending"
                     rank_method = st.selectbox("Method for Ties:", ['average', 'min', 'max', 'first', 'dense'], key=f"rank_method_{action}_t1")
                     new_rank_col_name = st.text_input("New Rank Column Name:", f"{rank_col}_rank", key=f"rank_newname_{action}_t1")
                     if not new_rank_col_name: new_rank_col_name = f"{rank_col}_rank"
                     code_string = f"result_data = df.copy()\nrank_col_numeric = pd.to_numeric(result_data['{rank_col}'], errors='coerce')\nresult_data['{new_rank_col_name}'] = rank_col_numeric.rank(method='{rank_method}', ascending={rank_ascending})"
                     action_configured = True

            # === Visualization ===
            elif action == "Plot Histogram (Numeric)":
                hist_col = st.selectbox("Select Numeric Column:", wiz_numeric_cols, key=f"hist_col_{action}_t1")
                if hist_col:
                    bins = st.slider("Number of Bins:", 5, 100, 20, key=f"hist_bins_{action}_t1")
                    kde = st.checkbox("Show Density Curve (KDE)", value=True, key=f"hist_kde_{action}_t1")
                    code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nsns.histplot(data=df, x='{hist_col}', bins={bins}, kde={kde}, ax=ax)\nax.set_title('Histogram of {hist_col}')\nplt.tight_layout()\nresult_data = fig"""
                    action_configured = True
            elif action == "Plot Density Plot (Numeric)":
                dens_col = st.selectbox("Select Numeric Column:", wiz_numeric_cols, key=f"dens_col_{action}_t1")
                if dens_col:
                    shade = st.checkbox("Shade Area", value=True, key=f"dens_shade_{action}_t1")
                    code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nsns.kdeplot(data=df, x='{dens_col}', fill={shade}, ax=ax)\nax.set_title('Density Plot of {dens_col}')\nplt.tight_layout()\nresult_data = fig"""
                    action_configured = True
            elif action == "Plot Count Plot (Categorical)":
                count_col = st.selectbox("Select Categorical Column:", wiz_categorical_cols, key=f"count_col_{action}_t1")
                if count_col:
                    top_n_check = st.checkbox("Show Only Top N?", value=True, key=f"count_topn_check_{action}_t1")
                    top_n = 20
                    if top_n_check: top_n = st.slider("Top N Categories:", 5, 50, 20, key=f"count_topn_slider_{action}_t1")
                    code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nuse_top_n = {top_n_check}\nn = {top_n}\ncol = '{count_col}'\nif col in df.columns:\n    if use_top_n and df[col].nunique() > n:\n        plot_order = df[col].value_counts().nlargest(n).index\n        plot_data = df[df[col].isin(plot_order)]\n        sns.countplot(y=col, data=plot_data, order=plot_order, ax=ax)\n        ax.set_title(f'Top {{n}} Counts for {{col}}')\n    else:\n        sns.countplot(y=col, data=df, order=df[col].value_counts().index, ax=ax)\n        ax.set_title(f'Counts for {{col}}')\n    plt.tight_layout()\n    result_data = fig\nelse:\n    print(f"Error: Column '{{col}}' not found.")\n    result_data = None"""
                    action_configured = True
            elif action == "Plot Bar Chart (Aggregated)":
                 st.info("Plot from original data or previous result (if DataFrame).")
                 prev_result = st.session_state.get('current_action_result_display', None)
                 df_options_bar = {"Use Original DataFrame": df}
                 if isinstance(prev_result, pd.DataFrame): df_options_bar["Use Previous Wizard Result"] = prev_result
                 selected_df_key_bar = st.radio("Use Data From:", list(df_options_bar.keys()), horizontal=True, key=f"bar_df_source_{action}_t1")
                 bar_df_source = df_options_bar[selected_df_key_bar]
                 if isinstance(bar_df_source, pd.DataFrame) and len(bar_df_source.columns) >= 2:
                     num_cols_bar, cat_cols_bar, _ = get_column_types(bar_df_source)
                     x_col_bar = st.selectbox("Categorical (X-axis):", cat_cols_bar, key=f"bar_x_{action}_t1")
                     y_col_bar = st.selectbox("Numeric (Y-axis):", num_cols_bar, key=f"bar_y_{action}_t1")
                     if x_col_bar and y_col_bar:
                         code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nsns.barplot(x='{x_col_bar}', y='{y_col_bar}', data=plot_source_df, ax=ax)\nax.set_title('Bar Chart: {y_col_bar} by {x_col_bar}')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nresult_data = fig"""
                         action_configured = True # Need to pass bar_df_source later
                     elif not x_col_bar or not y_col_bar: st.caption("Select X and Y.")
                 else: st.warning("Selected source not suitable (needs cat/num cols).")
            elif action == "Plot Line Chart":
                 st.info("Select X (Date/Sequence) and Y (Numeric).")
                 x_col_line = st.selectbox("X-axis Column:", all_cols, key=f"line_x_{action}_t1")
                 y_col_line = st.selectbox("Y-axis Column (Numeric):", wiz_numeric_cols, key=f"line_y_{action}_t1")
                 hue_col_line = st.selectbox("Group Lines By (Cat - Opt.):", [None] + wiz_categorical_cols, key=f"line_hue_{action}_t1")
                 if x_col_line and y_col_line:
                     hue_param = f", hue='{hue_col_line}'" if hue_col_line else ""
                     code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nplot_df_line = df.copy()\ntry:\n    x_col = '{x_col_line}'\n    if pd.api.types.is_datetime64_any_dtype(plot_df_line[x_col]) or ('date' in x_col.lower() or 'time' in x_col.lower()):\n        plot_df_line[x_col] = pd.to_datetime(plot_df_line[x_col], errors='coerce')\n    if not plot_df_line[x_col].isnull().all(): plot_df_line = plot_df_line.sort_values(by=x_col)\nexcept Exception as e:\n    print(f"Warning: Could not convert/sort X-axis '{x_col_line}': {{e}}")\nsns.lineplot(x='{x_col_line}', y='{y_col_line}'{hue_param}, data=plot_df_line, ax=ax)\nax.set_title('Line Chart: {y_col_line} over {x_col_line}')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nresult_data = fig"""
                     action_configured = True
            elif action == "Plot Scatter Plot":
                 st.info("Relationship between two numeric variables.")
                 x_col_scatter = st.selectbox("X-axis (Numeric):", wiz_numeric_cols, key=f"scatter_x_{action}_t1")
                 y_col_scatter = st.selectbox("Y-axis (Numeric):", wiz_numeric_cols, key=f"scatter_y_{action}_t1")
                 hue_col_scatter = st.selectbox("Color By (Cat - Opt.):", [None] + wiz_categorical_cols, key=f"scatter_hue_{action}_t1")
                 size_col_scatter = st.selectbox("Size By (Num - Opt.):", [None] + wiz_numeric_cols, key=f"scatter_size_{action}_t1")
                 if x_col_scatter and y_col_scatter:
                     hue_param = f", hue='{hue_col_scatter}'" if hue_col_scatter else ""
                     size_param = f", size='{size_col_scatter}'" if size_col_scatter else ""
                     code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nsns.scatterplot(x='{x_col_scatter}', y='{y_col_scatter}'{hue_param}{size_param}, data=df, ax=ax)\nax.set_title('Scatter Plot: {y_col_scatter} vs {x_col_scatter}')\nplt.tight_layout()\nresult_data = fig"""
                     action_configured = True
            elif action == "Plot Box Plot (Numeric vs Cat)":
                 st.info("Compare numeric distribution across categories.")
                 x_col_box = st.selectbox("Categorical (X-axis):", wiz_categorical_cols, key=f"box_x_{action}_t1")
                 y_col_box = st.selectbox("Numeric (Y-axis):", wiz_numeric_cols, key=f"box_y_{action}_t1")
                 if x_col_box and y_col_box:
                     limit_cats_box = st.checkbox("Limit Categories?", value=True, key=f"box_limit_{action}_t1")
                     top_n_box = 15
                     if limit_cats_box: top_n_box = st.slider("Max Categories:", 5, 50, 15, key=f"box_topn_{action}_t1")
                     code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nlimit_cats = {limit_cats_box}\ntop_n = {top_n_box}\nx_col = '{x_col_box}'\ny_col = '{y_col_box}'\nplot_data_box = df.copy()\nplot_data_box[y_col] = pd.to_numeric(plot_data_box[y_col], errors='coerce')\nplot_data_box.dropna(subset=[x_col, y_col], inplace=True)\nif limit_cats and plot_data_box[x_col].nunique() > top_n:\n    plot_order_box = plot_data_box[x_col].value_counts().nlargest(top_n).index\n    plot_data_box = plot_data_box[plot_data_box[x_col].isin(plot_order_box)]\n    sns.boxplot(x=x_col, y=y_col, data=plot_data_box, order=plot_order_box, ax=ax)\n    ax.set_title(f'Box Plot: {{y_col}} by Top {{top_n}} {{x_col}}')\nelse:\n    plot_order_box = plot_data_box[x_col].value_counts().index\n    sns.boxplot(x=x_col, y=y_col, data=plot_data_box, order=plot_order_box, ax=ax)\n    ax.set_title(f'Box Plot: {{y_col}} by {{x_col}}')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nresult_data = fig"""
                     action_configured = True
            elif action == "Plot Correlation Heatmap (Numeric)":
                 st.info("Correlation between numeric columns.")
                 default_corr_cols = wiz_numeric_cols[:min(len(wiz_numeric_cols), 15)]
                 corr_cols = st.multiselect("Select Numeric Columns (Min 2):", wiz_numeric_cols, default=default_corr_cols, key=f"corr_cols_{action}_t1")
                 if len(corr_cols) >= 2:
                     corr_method = st.selectbox("Correlation Method:", ['pearson', 'kendall', 'spearman'], key=f"corr_method_{action}_t1")
                     code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\ncorr_df = df[{corr_cols}].copy()\nfor col in {corr_cols}: corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')\ncorr_matrix = corr_df.corr(method='{corr_method}')\nfig, ax = plt.subplots(figsize=(max(6, len({corr_cols})*0.8), max(5, len({corr_cols})*0.7)))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)\nax.set_title('Correlation Heatmap ({corr_method.capitalize()})')\nplt.tight_layout()\nresult_data = fig"""
                     action_configured = True
                 else: st.warning("Select at least two numeric columns.")
            # --- End of Wizard Action Implementation Block ---
        except Exception as e:
            st.error(f"Error configuring action '{action}': {e}")
            st.error(traceback.format_exc())
            code_string = "# Error during configuration"
            action_configured = False

        # --- Trigger Execution (Wizard) ---
        st.subheader("3. Apply Action")
        apply_col1, apply_col2 = st.columns([1, 3])
        with apply_col1:
            apply_button_pressed = st.button(f"Apply: {action}", key=f"apply_{action}_t1", use_container_width=True, disabled=not action_configured)
        if not action_configured and action and not action.startswith("---"):
             with apply_col2: st.caption("👈 Configure all options first.")

        if apply_button_pressed and action_configured:
            if code_string and not code_string.startswith("# Error"):
                st.session_state.generated_code = code_string # Store code
                # Clear previous results/flags before execution
                st.session_state.result_type = None
                st.session_state.show_profile_report = False
                st.session_state.show_sweetviz_report = False
                st.session_state.current_action_result_display = None
                report_placeholder_tab1.empty()

                with st.spinner(f"Applying: {action}..."):
                    local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'result_data': None}
                    if action == "Plot Bar Chart (Aggregated)" and isinstance(bar_df_source, pd.DataFrame):
                        local_vars['plot_source_df'] = bar_df_source

                    try:
                        exec(code_string, {'pd': pd, 'np': np, 'plt': plt, 'sns': sns}, local_vars)
                        res = local_vars.get('result_data')
                        current_result_to_display, current_result_type = None, None
                        if isinstance(res, pd.DataFrame): current_result_type, current_result_to_display = 'dataframe', res
                        elif isinstance(res, pd.Series): current_result_type, current_result_to_display = 'dataframe', res.to_frame()
                        elif isinstance(res, plt.Figure): current_result_type, current_result_to_display = 'plot', res
                        elif res is not None: current_result_type, current_result_to_display = 'scalar_text', res

                        st.session_state.current_action_result_display = current_result_to_display
                        st.session_state.result_type = current_result_type

                        if current_result_to_display is not None: st.success("Action applied!")
                        else: st.warning("Action ran, but no displayable result produced.")
                        st.rerun() # Rerun to display result below

                    except Exception as e:
                        st.error(f"Error executing action: {e}")
                        st.error(traceback.format_exc())
                        st.session_state.current_action_result_display = None
                        with st.expander("Show Failing Code"): st.code(st.session_state.generated_code, language='python')
            elif not action_configured:
                 with apply_col2: st.warning("Configure options first.")

        # --- Display Wizard Results ---
        st.subheader("📊 Wizard Results")
        result_display_placeholder_tab1 = st.empty()
        with result_display_placeholder_tab1.container():
            current_result = st.session_state.get('current_action_result_display', None)
            current_type = st.session_state.get('result_type', None)
            if current_result is not None:
                try:
                    if current_type == 'dataframe':
                        st.dataframe(current_result)
                        res_df_dl = current_result
                        col1_dl, col2_dl = st.columns(2)
                        with col1_dl: st.download_button("Download CSV", convert_df_to_csv(res_df_dl), "wizard_result.csv", "text/csv", use_container_width=True, key="dl_csv_t1")
                        with col2_dl: st.download_button("Download Excel", convert_df_to_excel(res_df_dl), "wizard_result.xlsx", "application/...", use_container_width=True, key="dl_excel_t1")
                    elif current_type == 'plot':
                        if isinstance(current_result, plt.Figure):
                             st.pyplot(current_result)
                             plot_bytes = save_plot_to_bytes(current_result) # Pass the fig object
                             if plot_bytes: st.download_button("Download Plot PNG", plot_bytes, "wizard_plot.png", "image/png", use_container_width=True, key="dl_plot_t1")
                             else: st.warning("Could not generate plot file.")
                             plt.close(current_result) # Close figure after displaying and saving
                        else: st.error("Invalid plot object.")
                    elif current_type == 'scalar_text':
                         st.write("Result:")
                         if isinstance(current_result, (dict, list)): st.json(current_result)
                         else: st.write(current_result)
                         try:
                              scalar_data = str(current_result).encode('utf-8')
                              st.download_button("Download Text", scalar_data, "wizard_result.txt", "text/plain", use_container_width=True, key="dl_scalar_t1")
                         except Exception as scalar_dl_err: st.warning(f"Could not generate text file: {scalar_dl_err}")

                    if st.session_state.generated_code:
                         with st.expander("Show Python Code for This Result"):
                              st.code(st.session_state.generated_code, language='python')
                except Exception as display_err:
                     st.error(f"Error displaying/downloading result: {display_err}")
                     st.error(traceback.format_exc())
            else:
                 if not st.session_state.get('show_profile_report', False) and not st.session_state.get('show_sweetviz_report', False):
                     st.caption("Apply an action from the wizard above to see results here.")


    # =============================================
    # == Tab 2: ML Data Readiness Assessment ======
    # =============================================
    with tab2:
        st.header("🤖 ML Data Readiness Assessment")
        st.markdown("Analyze data suitability for common Machine Learning tasks.")

        potential_target = st.selectbox("1. Select Potential Target Variable:", [None] + all_cols, key="ml_target_select_t2")

        if potential_target:
            st.subheader(f"Analysis for Target: `{potential_target}`")
            target_series = df[potential_target]
            problem_type = None
            problem_details = ""

            # --- Target Analysis ---
            st.markdown("**Target Variable Insights:**")
            target_dtype = target_series.dtype
            target_n_unique = target_series.nunique()

            if pd.api.types.is_numeric_dtype(target_dtype):
                if target_n_unique / len(target_series) > 0.1 and target_n_unique > 30: # Heuristic for continuous
                    problem_type = "Regression"
                    problem_details = f"Numeric, high unique values ({target_n_unique}). Likely **Regression**."
                    st.metric("Suggested Problem Type", "Regression", delta=f"{target_n_unique} unique values", delta_color="off")
                else: # Low cardinality numeric or integer
                    problem_type = "Classification"
                    problem_details = f"Numeric, low unique values ({target_n_unique}). Could be **Classification** (or Ordinal Regression)."
                    st.metric("Suggested Problem Type", "Classification", delta=f"{target_n_unique} unique values", delta_color="off")
                    st.info("Consider if the numeric values represent distinct categories.")
            elif pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype) or pd.api.types.is_bool_dtype(target_dtype):
                problem_type = "Classification"
                if target_n_unique == 2:
                    problem_details = f"{target_dtype}, 2 unique values. Likely **Binary Classification**."
                    st.metric("Suggested Problem Type", "Binary Classification", delta="2 unique values", delta_color="off")
                else:
                    problem_details = f"{target_dtype}, {target_n_unique} unique values. Likely **Multiclass Classification**."
                    st.metric("Suggested Problem Type", "Multiclass Classification", delta=f"{target_n_unique} unique values", delta_color="off")
                if target_n_unique > 50:
                    st.warning(f"High number of classes ({target_n_unique}). May require special handling.")
            else:
                 problem_details = f"Type: {target_dtype}. Could not automatically determine ML problem type."
                 st.metric("Suggested Problem Type", "Unknown", delta=f"Type: {target_dtype}", delta_color="off")

            st.write(f"**Target Summary:** {problem_details}")
            if target_series.isnull().sum() > 0:
                 st.warning(f"Target variable '{potential_target}' has {target_series.isnull().sum()} missing values ({target_series.isnull().mean()*100:.2f}%). These rows usually need to be dropped for supervised ML.")

            st.divider()

            # --- Feature Analysis ---
            st.markdown("**Feature Analysis (Potential Issues):**")
            features_df = df.drop(columns=[potential_target])
            feature_issues = []

            # Missing Values
            missing_pct = features_df.isnull().mean() * 100
            high_missing = missing_pct[missing_pct > 30] # Threshold for high missing %
            if not high_missing.empty:
                 st.warning("High Missing Values (>30%):")
                 st.dataframe(high_missing.reset_index().rename(columns={'index':'Feature', 0:'Missing %'}))
                 feature_issues.append(f"High missing values in {len(high_missing)} feature(s).")

            # Cardinality (Categorical)
            high_cardinality_limit = 50
            high_card_features = []
            for col in categorical_cols:
                if col != potential_target and features_df[col].nunique() > high_cardinality_limit:
                    high_card_features.append({'Feature': col, 'Unique Values': features_df[col].nunique()})
            if high_card_features:
                 st.warning(f"High Cardinality Categorical Features (> {high_cardinality_limit} unique):")
                 st.dataframe(pd.DataFrame(high_card_features))
                 feature_issues.append(f"High cardinality in {len(high_card_features)} categorical feature(s). Requires careful encoding.")

             # Skewness (Numeric)
            skew_threshold = 1.0 # Absolute skewness threshold
            numeric_features = features_df.select_dtypes(include=np.number)
            skewness = numeric_features.apply(lambda x: skew(x.dropna())).abs()
            high_skew = skewness[skewness > skew_threshold]
            if not high_skew.empty:
                 st.warning(f"Highly Skewed Numeric Features (Abs Skewness > {skew_threshold}):")
                 st.dataframe(high_skew.reset_index().rename(columns={'index':'Feature', 0:'Skewness'}))
                 feature_issues.append(f"High skewness in {len(high_skew)} numeric feature(s). Consider transformations (log, sqrt).")

            # Near-Zero Variance (Constant/Near-Constant Features)
            near_zero_var_features = []
            for col in features_df.columns:
                # Check if mostly constant (e.g., > 98% same value)
                n_unique = features_df[col].nunique()
                if n_unique <= 1:
                     near_zero_var_features.append({'Feature': col, 'Reason': 'Constant (1 unique value)'})
                elif n_unique / len(features_df) < 0.01 : # Very low unique ratio
                     counts = features_df[col].value_counts(normalize=True)
                     if counts.iloc[0] > 0.98: # Most frequent value makes up > 98%
                         near_zero_var_features.append({'Feature': col, 'Reason': f'Near Constant (>98% is "{counts.index[0]}")'})
            if near_zero_var_features:
                 st.warning("Near Zero Variance Features (Mostly Constant):")
                 st.dataframe(pd.DataFrame(near_zero_var_features))
                 feature_issues.append(f"{len(near_zero_var_features)} feature(s) with very low variance. Often removed.")

             # Correlation with Target (if Regression)
            if problem_type == "Regression":
                 st.markdown("**Correlation with Target (Numeric Features):**")
                 try:
                     numeric_features_corr = df[numeric_cols].copy() # Use original df with target
                     # Ensure target is numeric for correlation calculation
                     numeric_features_corr[potential_target] = pd.to_numeric(numeric_features_corr[potential_target], errors='coerce')
                     numeric_features_corr = numeric_features_corr.dropna(subset=[potential_target]) # Drop rows where target is NA after conversion

                     correlations = numeric_features_corr.corrwith(numeric_features_corr[potential_target]).drop(potential_target).sort_values(ascending=False)
                     st.dataframe(correlations.reset_index().rename(columns={'index':'Feature', 0:'Correlation'}))
                     # Plot top/bottom N correlations
                     top_n_corr = 10
                     if len(correlations) > 1:
                          show_corr_plot = st.checkbox("Show Correlation Plot (Top/Bottom N)", value=True, key="corr_plot_t2")
                          if show_corr_plot:
                             fig_corr, ax_corr = plt.subplots(figsize=(8, max(4, min(len(correlations), top_n_corr*2)*0.3)))
                             plot_corr = pd.concat([correlations.head(top_n_corr), correlations.tail(top_n_corr)]).sort_values()
                             sns.barplot(x=plot_corr.values, y=plot_corr.index, ax=ax_corr, palette="vlag")
                             ax_corr.set_title(f'Top/Bottom {top_n_corr} Feature Correlations with {potential_target}')
                             ax_corr.set_xlabel("Pearson Correlation")
                             plt.tight_layout()
                             st.pyplot(fig_corr)
                             plt.close(fig_corr) # Close figure
                 except Exception as corr_err:
                     st.warning(f"Could not calculate correlations with target: {corr_err}")


            # --- Overall Summary ---
            st.divider()
            st.subheader("Overall ML Readiness Score:")
            score = "Good"
            if len(feature_issues) > 2: score = "Low"
            elif len(feature_issues) > 0: score = "Moderate"

            if score == "Good": st.success(f"**{score}**")
            elif score == "Moderate": st.warning(f"**{score}**")
            else: st.error(f"**{score}**")

            if feature_issues:
                 st.markdown("**Potential Issues Found:**")
                 for issue in feature_issues: st.markdown(f"- {issue}")
            else:
                 st.markdown("No major data quality/suitability issues automatically detected (based on heuristics). Always perform deeper EDA.")

        else:
             st.info("👆 Select a potential target variable above to start the assessment.")

    # =============================================
    # == Tab 3: ML Preprocessing Simulation =======
    # =============================================
    with tab3:
        st.header("🛠️ ML Preprocessing Simulation")
        st.markdown("Configure standard preprocessing steps and generate scikit-learn code.")

        # --- Feature & Target Selection ---
        st.subheader("1. Select Target and Features")
        target_prep = st.selectbox("Target Variable (ignored by preprocessing):", [None] + all_cols, key="prep_target_t3")
        if target_prep:
             available_features = [col for col in all_cols if col != target_prep]
             selected_features = st.multiselect("Features to Preprocess:", available_features, default=available_features, key="prep_features_t3")
        else:
             selected_features = st.multiselect("Features to Preprocess:", all_cols, default=all_cols, key="prep_features_t3_notarget")

        # Filter column types based on selected features
        sel_num_cols = [c for c in numeric_cols if c in selected_features]
        sel_cat_cols = [c for c in categorical_cols if c in selected_features]
        sel_dt_cols = [c for c in datetime_cols if c in selected_features] # Datetime cols often dropped or engineered
        sel_text_cols = [c for c in text_cols if c in selected_features] # Text cols need specific vectorizers

        if not selected_features:
            st.warning("Select features to preprocess.")
        else:
            st.subheader("2. Configure Preprocessing Steps")
            prep_col1, prep_col2 = st.columns(2)

            # --- Numeric Preprocessing ---
            with prep_col1:
                st.markdown("**Numeric Features**")
                if not sel_num_cols: st.caption("No numeric features selected.")
                else:
                    num_imputation = st.selectbox("Imputation:", ["None", "Mean", "Median", "Constant (0)"], key="prep_num_impute_t3")
                    num_scaling = st.selectbox("Scaling:", ["None", "StandardScaler", "MinMaxScaler"], key="prep_num_scale_t3")

            # --- Categorical Preprocessing ---
            with prep_col2:
                st.markdown("**Categorical Features**")
                if not sel_cat_cols: st.caption("No categorical features selected.")
                else:
                    cat_imputation = st.selectbox("Imputation:", ["None", "Mode (Most Frequent)", "Constant ('missing')"], key="prep_cat_impute_t3")
                    cat_encoding = st.selectbox("Encoding:", ["None", "OneHotEncoder", "OrdinalEncoder (Use Cautiously)"], key="prep_cat_encode_t3")
                    handle_unknown = 'ignore' if cat_encoding == 'OneHotEncoder' else None # OHE param
                    if cat_encoding == "OneHotEncoder":
                         handle_unknown = st.radio("Handle Unknown Categories (OHE):", ['ignore', 'error'], index=0, horizontal=True, key="prep_ohe_unknown_t3")


            st.markdown("**Other Feature Types** (Will be Dropped by Default)")
            if sel_dt_cols: st.caption(f"- Datetime: {sel_dt_cols}")
            if sel_text_cols: st.caption(f"- Text (heuristic): {sel_text_cols}")
            st.info("Standard pipelines often drop datetime/text unless specific transformers (e.g., TF-IDF) are added manually.")

            # --- Generate Button ---
            st.divider()
            if st.button("Generate Preprocessing Code & Preview", key="prep_generate_t3"):
                with st.spinner("Generating pipeline and preview..."):
                    # --- Build Numeric Pipeline ---
                    num_steps = []
                    if num_imputation != "None":
                        strategy = 'mean' if num_imputation == "Mean" else 'median' if num_imputation == "Median" else 'constant'
                        fill_value = 0 if num_imputation == "Constant (0)" else None
                        num_steps.append(('imputer', SimpleImputer(strategy=strategy, fill_value=fill_value)))
                    if num_scaling == "StandardScaler":
                        num_steps.append(('scaler', StandardScaler()))
                    elif num_scaling == "MinMaxScaler":
                        num_steps.append(('scaler', MinMaxScaler()))

                    numeric_pipeline = Pipeline(steps=num_steps) if num_steps else 'drop'

                    # --- Build Categorical Pipeline ---
                    cat_steps = []
                    if cat_imputation != "None":
                        strategy = 'most_frequent' if cat_imputation == "Mode (Most Frequent)" else 'constant'
                        fill_value = 'missing' if cat_imputation == "Constant ('missing')" else None
                        cat_steps.append(('imputer', SimpleImputer(strategy=strategy, fill_value=fill_value)))
                    if cat_encoding == "OneHotEncoder":
                         # Add handle_unknown based on user choice
                        cat_steps.append(('onehot', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))) # sparse=False easier for preview
                    elif cat_encoding == "OrdinalEncoder (Use Cautiously)":
                         cat_steps.append(('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))) # Handle potential unknowns

                    categorical_pipeline = Pipeline(steps=cat_steps) if cat_steps else 'drop'

                    # --- Create ColumnTransformer ---
                    transformers_list = []
                    if numeric_pipeline != 'drop' and sel_num_cols:
                         transformers_list.append(('num', numeric_pipeline, sel_num_cols))
                    if categorical_pipeline != 'drop' and sel_cat_cols:
                         transformers_list.append(('cat', categorical_pipeline, sel_cat_cols))

                    # Determine remainder action (drop other columns like datetime/text)
                    preprocessor = ColumnTransformer(
                        transformers=transformers_list,
                        remainder='drop' # Drop columns not specified (datetime, text etc.)
                    )

                    # --- Generate Code String ---
                    code_gen = f"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

# Define features
numeric_features = {sel_num_cols}
categorical_features = {sel_cat_cols}

# Define preprocessing steps
numeric_transformer = {repr(numeric_pipeline)} # Use repr for Pipeline/string

categorical_transformer = {repr(categorical_pipeline)} # Use repr for Pipeline/string

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Or 'passthrough' if needed
)

# Example usage:
# Assuming 'df' is your DataFrame and 'selected_features' is a list of columns
# X = df[selected_features]
# X_processed = preprocessor.fit_transform(X)

# To get feature names after transformation (especially after OneHotEncoder):
# try:
#     feature_names_out = preprocessor.get_feature_names_out()
# except Exception as e:
#     print(f"Could not get feature names automatically: {{e}}")
#     feature_names_out = None

# processed_df = pd.DataFrame(X_processed, columns=feature_names_out, index=X.index)

"""
                    st.subheader("Generated Scikit-learn Code:")
                    st.code(code_gen, language='python')

                    # --- Preview Transformed Data (on sample) ---
                    st.subheader("Preview of Transformed Data (Sample)")
                    try:
                        sample_df = df[selected_features].sample(n=min(100, len(df)), random_state=42)
                        # Fit and transform the sample
                        transformed_sample = preprocessor.fit_transform(sample_df)

                        # Try to get feature names
                        try:
                             feature_names = preprocessor.get_feature_names_out()
                             transformed_df_preview = pd.DataFrame(transformed_sample, columns=feature_names, index=sample_df.index)
                        except Exception:
                             st.warning("Could not automatically get feature names after transformation. Displaying as numpy array.")
                             transformed_df_preview = pd.DataFrame(transformed_sample) # Show as numpy array if names fail

                        st.dataframe(transformed_df_preview.head())
                        st.caption(f"Showing head() of transformed data based on a sample of {len(sample_df)} rows.")

                    except Exception as e:
                        st.error(f"Error applying preprocessing pipeline to sample data: {e}")
                        st.error(traceback.format_exc())
                        st.info("This can happen if incompatible steps are chosen (e.g., scaling non-numeric data due to imputation issues) or if data has unexpected values.")

    # =============================================
    # == Tab 4: XAI Feature Insights ==============
    # =============================================
    with tab4:
        st.header("💡 Explainable AI (XAI) Feature Insights")
        st.markdown("Explore relationships between features and a potential target variable.")

        target_xai = st.selectbox("1. Select Target Variable:", [None] + all_cols, key="xai_target_t4")

        if target_xai:
            st.subheader(f"Insights Relative to Target: `{target_xai}`")
            df_xai = df.copy() # Work on a copy
            target_series_xai = df_xai[target_xai]
            features_xai = df_xai.drop(columns=[target_xai])

            # Identify target type
            target_dtype_xai = target_series_xai.dtype
            target_n_unique_xai = target_series_xai.nunique()
            is_target_numeric_cont = pd.api.types.is_numeric_dtype(target_dtype_xai) and target_n_unique_xai / len(target_series_xai) > 0.05 and target_n_unique_xai > 10
            is_target_classification = not is_target_numeric_cont or target_n_unique_xai <= 10 # Heuristic

            # Get feature types for XAI context
            num_cols_xai, cat_cols_xai, dt_cols_xai, txt_cols_xai = get_column_types(features_xai, include_text_guess=True)

            # --- Calculate Simple Feature Importance/Scores ---
            st.subheader("2. Feature Scores (Potential Importance)")
            top_n_features = 15 # Number of features to show in plots/tables
            scores_df = None
            try:
                # Prepare data for scoring (handle missing values simply for this)
                X_scores = features_xai.copy()
                y_scores = target_series_xai.copy()

                # Impute numeric features simply (median) for scoring robustness
                num_imputer_scores = SimpleImputer(strategy='median')
                if num_cols_xai:
                    X_scores[num_cols_xai] = num_imputer_scores.fit_transform(X_scores[num_cols_xai])

                # Impute/Encode categoricals simply (mode/ordinal) for scoring
                cat_imputer_scores = SimpleImputer(strategy='most_frequent')
                if cat_cols_xai:
                    X_scores[cat_cols_xai] = cat_imputer_scores.fit_transform(X_scores[cat_cols_xai])
                    # Use Ordinal Encoding for simplicity in scoring (though not ideal for modeling)
                    ordinal_encoder_scores = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_scores[cat_cols_xai] = ordinal_encoder_scores.fit_transform(X_scores[cat_cols_xai])

                # Drop other columns (datetime, text) for basic scoring
                cols_to_keep_scores = num_cols_xai + cat_cols_xai
                X_scores = X_scores[cols_to_keep_scores]

                # Drop rows where target is still NA (important after potential target conversion/initial state)
                not_na_target_idx = y_scores.notna()
                y_scores = y_scores[not_na_target_idx]
                X_scores = X_scores[not_na_target_idx]


                if is_target_numeric_cont and not y_scores.empty: # Regression Scoring
                    f_values, p_values = f_regression(X_scores, y_scores)
                    scores_df = pd.DataFrame({'Feature': cols_to_keep_scores, 'F-statistic': f_values, 'p-value': p_values})
                    scores_df = scores_df.sort_values('F-statistic', ascending=False).reset_index(drop=True)
                    st.write("Using F-statistic (Linear dependency): Higher is potentially more important.")

                elif is_target_classification and not y_scores.empty: # Classification Scoring
                    # Use f_classif (ANOVA F-value) for numeric features
                    scores_num = {}
                    if num_cols_xai:
                         f_vals_cls, p_vals_cls = f_classif(X_scores[num_cols_xai], y_scores)
                         scores_num = pd.DataFrame({'Feature': num_cols_xai, 'F-statistic': f_vals_cls, 'p-value': p_vals_cls})

                    # Use chi2 for categorical features (needs non-negative values, ordinal encoding is okay here)
                    scores_cat = {}
                    if cat_cols_xai:
                        # Ensure non-negative before chi2
                        X_cat_scores = X_scores[cat_cols_xai]
                        if (X_cat_scores < 0).any().any():
                            st.warning("Ordinal encoded features have negative values (likely from unknown handling), cannot use Chi2 directly. Skipping Chi2 scores.")
                        else:
                            chi2_vals, p_vals_chi2 = chi2(X_cat_scores, y_scores)
                            scores_cat = pd.DataFrame({'Feature': cat_cols_xai, 'Chi2 Stat': chi2_vals, 'p-value': p_vals_chi2})

                    # Combine scores (use different metrics)
                    scores_df_list = []
                    if isinstance(scores_num, pd.DataFrame): scores_df_list.append(scores_num.rename(columns={'F-statistic': 'Score', 'p-value':'p_value'}))
                    if isinstance(scores_cat, pd.DataFrame): scores_df_list.append(scores_cat.rename(columns={'Chi2 Stat': 'Score', 'p-value':'p_value'}))

                    if scores_df_list:
                        scores_df = pd.concat(scores_df_list).sort_values('Score', ascending=False).reset_index(drop=True)
                        st.write("Using F-statistic (Numeric) & Chi2 (Categorical): Higher score suggests stronger relationship.")
                    else:
                         st.warning("Could not calculate feature scores for classification.")

                if scores_df is not None:
                    st.dataframe(scores_df.head(top_n_features))
                else:
                     st.warning("Could not calculate feature scores (e.g., empty data after cleaning, all features dropped).")

            except Exception as score_err:
                 st.error(f"Error calculating feature scores: {score_err}")
                 st.error(traceback.format_exc())

            st.divider()

            # --- Visualization ---
            st.subheader(f"3. Visualizations vs Target `{target_xai}`")

            # Get top N features based on scores if available, otherwise just use first N numeric/cat
            top_num_features_xai = num_cols_xai
            top_cat_features_xai = cat_cols_xai
            if scores_df is not None:
                 top_features_scored = scores_df['Feature'].head(top_n_features * 2).tolist() # Get more for flexibility
                 top_num_features_xai = [f for f in num_cols_xai if f in top_features_scored][:top_n_features]
                 top_cat_features_xai = [f for f in cat_cols_xai if f in top_features_scored][:top_n_features]
                 if not top_num_features_xai: top_num_features_xai = num_cols_xai[:top_n_features] # Fallback
                 if not top_cat_features_xai: top_cat_features_xai = cat_cols_xai[:top_n_features] # Fallback

            # Ensure target is suitable for plotting (e.g., numeric for scatter Y)
            plot_df_xai = df[[target_xai] + top_num_features_xai + top_cat_features_xai].copy()
            if is_target_numeric_cont:
                 plot_df_xai[target_xai] = pd.to_numeric(plot_df_xai[target_xai], errors='coerce')
            plot_df_xai.dropna(subset=[target_xai], inplace=True) # Drop rows where target is NA for plotting

            if plot_df_xai.empty:
                 st.warning("No data available for plotting after handling missing target values.")
            else:
                # --- Regression Target Plots ---
                if is_target_numeric_cont:
                    st.markdown("**Numeric Features vs. Continuous Target**")
                    if top_num_features_xai:
                        selected_scatter_feat = st.selectbox("Select Numeric Feature for Scatter Plot:", top_num_features_xai, key="xai_scatter_feat_t4")
                        if selected_scatter_feat:
                            fig_scatter, ax_scatter = plt.subplots()
                            sns.scatterplot(data=plot_df_xai, x=selected_scatter_feat, y=target_xai, alpha=0.6, ax=ax_scatter)
                            ax_scatter.set_title(f'{selected_scatter_feat} vs {target_xai}')
                            plt.tight_layout()
                            st.pyplot(fig_scatter)
                            plt.close(fig_scatter)
                    else: st.caption("No numeric features found/selected.")

                    st.markdown("**Target Distribution across Categorical Features**")
                    if top_cat_features_xai:
                        selected_box_feat = st.selectbox("Select Categorical Feature for Box Plot:", top_cat_features_xai, key="xai_box_feat_reg_t4")
                        if selected_box_feat:
                             # Limit categories shown in box plot for clarity
                            n_unique_cat = plot_df_xai[selected_box_feat].nunique()
                            max_cats_box = 15
                            if n_unique_cat > max_cats_box:
                                 st.info(f"Too many categories ({n_unique_cat}). Showing top {max_cats_box} by count.")
                                 top_cats = plot_df_xai[selected_box_feat].value_counts().nlargest(max_cats_box).index
                                 plot_df_box = plot_df_xai[plot_df_xai[selected_box_feat].isin(top_cats)]
                                 order = top_cats
                            else:
                                 plot_df_box = plot_df_xai
                                 order = plot_df_box[selected_box_feat].value_counts().index # Order by count

                            fig_box, ax_box = plt.subplots(figsize=(10, 5))
                            sns.boxplot(data=plot_df_box, x=selected_box_feat, y=target_xai, order=order, ax=ax_box)
                            ax_box.set_title(f'{target_xai} Distribution by {selected_box_feat}')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_box)
                            plt.close(fig_box)
                    else: st.caption("No categorical features found/selected.")


                # --- Classification Target Plots ---
                elif is_target_classification:
                    st.markdown("**Numeric Features Distribution by Target Class**")
                    if top_num_features_xai:
                        selected_kde_feat = st.selectbox("Select Numeric Feature for Density Plot:", top_num_features_xai, key="xai_kde_feat_t4")
                        if selected_kde_feat:
                            fig_kde, ax_kde = plt.subplots()
                            sns.kdeplot(data=plot_df_xai, x=selected_kde_feat, hue=target_xai, fill=True, common_norm=False, alpha=0.5, ax=ax_kde)
                            ax_kde.set_title(f'{selected_kde_feat} Distribution by {target_xai}')
                            plt.tight_layout()
                            st.pyplot(fig_kde)
                            plt.close(fig_kde)
                    else: st.caption("No numeric features found/selected.")

                    st.markdown("**Categorical Features Distribution by Target Class**")
                    if top_cat_features_xai:
                        selected_count_feat = st.selectbox("Select Categorical Feature for Count Plot:", top_cat_features_xai, key="xai_count_feat_t4")
                        if selected_count_feat:
                            fig_count, ax_count = plt.subplots()
                            # Use dodge for multiple bars per category
                            sns.countplot(data=plot_df_xai, y=selected_count_feat, hue=target_xai, order=plot_df_xai[selected_count_feat].value_counts().index[:15], ax=ax_count) # Limit y categories
                            ax_count.set_title(f'{selected_count_feat} Counts by {target_xai} (Top 15 cats)')
                            plt.tight_layout()
                            st.pyplot(fig_count)
                            plt.close(fig_count)
                    else: st.caption("No categorical features found/selected.")

                    st.markdown("**Numeric Features vs. Target Class (Box Plot)**")
                    if top_num_features_xai:
                         selected_box_cls_feat = st.selectbox("Select Numeric Feature for Box Plot:", top_num_features_xai, key="xai_box_cls_feat_t4")
                         if selected_box_cls_feat:
                             fig_box_cls, ax_box_cls = plt.subplots(figsize=(8, 4))
                             sns.boxplot(data=plot_df_xai, x=target_xai, y=selected_box_cls_feat, ax=ax_box_cls)
                             ax_box_cls.set_title(f'{selected_box_cls_feat} Distribution by {target_xai}')
                             plt.xticks(rotation=45, ha='right')
                             plt.tight_layout()
                             st.pyplot(fig_box_cls)
                             plt.close(fig_box_cls)
                    else: st.caption("No numeric features found/selected.")

                else:
                    st.info("Target variable type not clearly identified as Regression or Classification for standard XAI plots.")

        else:
             st.info("👆 Select a target variable above to explore feature relationships.")


    # =============================================
    # == Tab 5: ML Model Selection Advisor ========
    # =============================================
    with tab5:
        st.header("🧠 ML Model Selection Advisor")
        st.markdown("Get heuristic suggestions for suitable scikit-learn models based on data characteristics.")

        # --- Data Characteristics Analysis ---
        st.subheader("1. Data Characteristics Summary")
        n_rows, n_cols = df.shape
        st.metric("Number of Rows", n_rows)
        st.metric("Number of Features (excluding target if selected)", n_cols-1 if target_xai else n_cols) # Use target from XAI tab context
        st.write(f"- Numeric Features: {len(numeric_cols)}")
        st.write(f"- Categorical Features: {len(categorical_cols)}")
        st.write(f"- Datetime Features: {len(datetime_cols)}")
        st.write(f"- Text Features (heuristic): {len(text_cols)}")

        # --- Problem Type (Re-evaluate or use from Readiness Tab) ---
        st.subheader("2. Assumed Problem Type")
        target_model_adv = st.selectbox("Confirm Target Variable (Influences Suggestions):", [None] + all_cols, key="model_target_t5")
        problem_type_model = "Unknown"
        advice_list = []

        if target_model_adv:
            target_series_model = df[target_model_adv]
            target_dtype_model = target_series_model.dtype
            target_n_unique_model = target_series_model.nunique()

            if pd.api.types.is_numeric_dtype(target_dtype_model) and target_n_unique_model / n_rows > 0.05 and target_n_unique_model > 10:
                problem_type_model = "Regression"
                st.info("Based on the target, this looks like a **Regression** problem.")
                advice_list = [
                    ("Linear Regression / Ridge / Lasso", "Good baselines, handle linear relationships. Ridge/Lasso add regularization.", "https://scikit-learn.org/stable/modules/linear_model.html"),
                    ("Random Forest Regressor", "Powerful, handles non-linearities, less sensitive to scaling.", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"),
                    ("Gradient Boosting Regressor (e.g., XGBoost, LightGBM)", "Often top performance, can require tuning.", "https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting"),
                    ("Support Vector Regressor (SVR)", "Effective in high dimensions, kernel-based.", "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html")
                ]
                if n_rows > 100000: advice_list.append(("Consider SGDRegressor for very large datasets", "Stochastic Gradient Descent efficient for large scale.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html"))


            elif target_n_unique_model is not None: # Check for None if target not selected
                problem_type_model = "Classification"
                if target_n_unique_model == 2:
                    st.info("Based on the target, this looks like a **Binary Classification** problem.")
                else:
                    st.info(f"Based on the target, this looks like a **Multiclass Classification** problem ({target_n_unique_model} classes).")

                advice_list = [
                    ("Logistic Regression", "Good linear baseline, interpretable.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"),
                    ("Random Forest Classifier", "Powerful, handles non-linearities, less sensitive to scaling.", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
                    ("Gradient Boosting Classifier (e.g., XGBoost, LightGBM)", "Often top performance, can require tuning.", "https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting"),
                    ("Support Vector Classifier (SVC)", "Effective in high dimensions, kernel-based.", "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html")
                ]
                if len(text_cols) > 0:
                     advice_list.append(("Naive Bayes (MultinomialNB/ComplementNB)", "Good for text classification.", "https://scikit-learn.org/stable/modules/naive_bayes.html"))
                if n_rows > 100000: advice_list.append(("Consider SGDClassifier for very large datasets", "Stochastic Gradient Descent efficient for large scale.", "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html"))
                if problem_type_model == "Multiclass Classification":
                     st.caption("Ensure chosen models support multiclass directly or use appropriate wrappers (e.g., OneVsRestClassifier). Most listed above do.")

            else:
                st.warning("Could not determine problem type. Please select a target variable.")

        else:
             st.info("👆 Select a target variable to get model suggestions.")


        # --- Display Recommendations ---
        st.subheader("3. Model Suggestions")
        if advice_list:
            for name, desc, link in advice_list:
                st.markdown(f"**{name}**")
                st.markdown(f"- *Description:* {desc}")
                st.markdown(f"- *Scikit-learn Docs:* [{link}]({link})")
                st.markdown("---")
            st.success("These are starting points. Always experiment and evaluate models based on your specific goals and data.")
        elif target_model_adv:
             st.warning("No specific model suggestions generated based on the analyzed characteristics.")
        else:
             st.caption("Select a target variable above.")


# --- Handling Initial State (No DataFrame Loaded) ---
elif st.session_state.data_source is None:
    st.info("👈 Please choose a data source from the sidebar to get started.")
else:
    # If a source is selected but df is None (e.g., upload pending, DB fetch failed/not run)
    if st.session_state.data_source == 'Upload File (CSV/Excel)':
         st.info("👈 Please upload a CSV or Excel file using the sidebar.")
    elif st.session_state.data_source == 'Connect to Database':
         # Check if DB config failed early
         if not DB_SERVERS and os.path.exists(DB_CONFIG_FILE): # Config file exists but failed to load
              st.error(f"Failed to load database configurations from '{DB_CONFIG_FILE}'. Please check the file format and content.")
         elif not DB_SERVERS and not os.path.exists(DB_CONFIG_FILE): # Config file missing
              st.warning(f"Database configuration file '{DB_CONFIG_FILE}' not found. Cannot establish database connections.")
         else: # Config loaded, but data not fetched yet
              st.info("👈 Please complete the database connection steps and fetch data using the sidebar.")
