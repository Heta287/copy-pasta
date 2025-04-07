# Final Complete app.py (with DB Config File, Wizard, EDA Reports, and ML 'Lite' Tabs)

import streamlit as st
import pandas as pd
import pyodbc
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ydata_profiling import ProfileReport
import sweetviz as sv
import streamlit.components.v1 as components
import os
import json
import traceback
import math # For calculations like number of bins

# Imports for ML Lite Tabs
from scipy.stats import skew
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler # For preparing data for feature scores
from sklearn.impute import SimpleImputer

# --- Constants ---
DB_CONFIG_FILE = "db_config.json"
DEFAULT_ODBC_DRIVER = "{ODBC Driver 17 for SQL Server}" # Change if needed
DEFAULT_TRUSTED_CONNECTION = "yes" # Change to "no" if using UID/PWD by default
# DEFAULT_UID = "YOUR_DEFAULT_USERNAME" # Optional
# DEFAULT_PWD = "YOUR_DEFAULT_PASSWORD" # Optional

# --- Load DB Configuration ---
def load_db_config(config_file):
    """Loads DB server names and addresses from a JSON file."""
    db_servers = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                db_servers = json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error: Could not decode '{config_file}'. Please ensure it's valid JSON.")
        except Exception as e:
            st.error(f"Error reading DB config file '{config_file}': {e}")
    else:
        st.warning(f"'{config_file}' not found. Database connection options will be unavailable. Please create the file with server_name: server_address pairs.")
    return db_servers

DB_SERVERS = load_db_config(DB_CONFIG_FILE)

# --- Action Definitions (for Wizard) ---
ACTION_CATEGORIES = {
    "Basic Info": ["Show Shape", "Show Columns & Types", "Show Basic Statistics (Describe)", "Show Missing Values", "Show Unique Values (Categorical)", "Count Values (Categorical)"],
    "Data Cleaning": ["Drop Columns", "Rename Columns", "Handle Missing Data", "Drop Duplicate Rows", "Change Data Type", "String Manipulation (Trim, Case, Replace)", "Extract from Text (Regex)", "Date Component Extraction (Year, Month...)"],
    "Data Transformation": ["Filter Data", "Sort Data", "Select Columns", "Create Calculated Column (Basic Arithmetic)", "Bin Numeric Data (Cut)", "One-Hot Encode Categorical Column", "Pivot (Simple)" , "Melt (Unpivot)"],
    "Aggregation & Analysis": ["Calculate Single Aggregation (Sum, Mean...)", "Group By & Aggregate", "Calculate Rolling Window Statistics", "Calculate Cumulative Statistics", "Rank Data"],
    "Visualization": ["Plot Histogram (Numeric)", "Plot Density Plot (Numeric)", "Plot Count Plot (Categorical)", "Plot Bar Chart (Aggregated)", "Plot Line Chart", "Plot Scatter Plot", "Plot Box Plot (Numeric vs Cat)", "Plot Correlation Heatmap (Numeric)"]
}

# --- Helper Functions ---

@st.cache_data(show_spinner="Connecting to database to fetch names...")
def get_databases(server_name):
    if not server_name or server_name not in DB_SERVERS: return []
    server_address = DB_SERVERS[server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no': conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    databases = []
    try:
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}
        with pyodbc.connect(conn_str, timeout=10, attrs_before=conn_attrs) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');")
            databases = [row.name for row in cursor.fetchall()]
    except pyodbc.Error as ex: st.error(f"DB Connect Error (List DBs): {ex}")
    except Exception as e: st.error(f"Unexpected Error (List DBs): {e}")
    return databases

@st.cache_data(show_spinner="Fetching table names...")
def get_tables(server_name, db_name):
    if not server_name or not db_name or server_name not in DB_SERVERS: return []
    server_address = DB_SERVERS[server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};DATABASE={db_name};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no': conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    tables = []
    try:
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}
        with pyodbc.connect(conn_str, timeout=10, attrs_before=conn_attrs) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME")
            tables = [f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}" for row in cursor.fetchall()]
    except pyodbc.Error as ex: st.error(f"DB Connect Error (List Tables): {ex}")
    except Exception as e: st.error(f"Unexpected Error (List Tables): {e}")
    return tables

@st.cache_data(show_spinner="Fetching data from database...")
def fetch_data_from_db(_server_name, _db_name, _sql_query):
    df = pd.DataFrame()
    if not _server_name or not _db_name or _server_name not in DB_SERVERS:
        st.error("Invalid server selection.") ; return df
    server_address = DB_SERVERS[_server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};DATABASE={_db_name};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no': conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    try:
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 15, pyodbc.SQL_ATTR_CONNECTION_TIMEOUT: 60}
        with pyodbc.connect(conn_str, timeout=60, attrs_before=conn_attrs) as conn:
            df = pd.read_sql(_sql_query, conn)
            if not df.empty: df = df.convert_dtypes()
    except pyodbc.Error as ex: st.error(f"DB Query Error: {ex}"); st.code(_sql_query, language='sql')
    except Exception as e: st.error(f"Data Fetching Error: {e}"); st.error(traceback.format_exc())
    return df

@st.cache_data
def load_file_data(uploaded_file):
    df = None
    try:
        file_name = uploaded_file.name
        st.info(f"Loading file: '{file_name}'...")
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            selected_sheet = sheet_names[0] # Default to first
            if len(sheet_names) > 1:
                # Use session state to remember sheet selection? Or keep simple for now? Let's keep simple.
                 selected_sheet = st.selectbox(f"Select sheet from '{file_name}':", sheet_names, key=f"sheet_{file_name}")
            if selected_sheet:
                df = excel_file.parse(selected_sheet)
            else: st.error(f"No sheets found or selected in '{file_name}'.")
        else: st.error("Unsupported file format.")

        if df is not None and not df.empty: df = df.convert_dtypes()
    except Exception as e: st.error(f"Error processing file: {e}"); st.error(traceback.format_exc()); df = None
    return df

# Refined get_column_types
def get_column_types(df):
    if df is None: return [], [], []
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        bool_cols = df.select_dtypes(include=['boolean', 'bool']).columns.tolist() # Include both pandas and numpy bool
        datetime_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Combine bool into numeric for ML simplicity later if desired, or keep separate
        # Let's keep bool separate for now for clarity
        numeric_cols = list(set(numeric_cols) - set(bool_cols)) # Remove bool from numeric if pandas includes it

        # Refine categorical
        potential_cat_cols = categorical_cols.copy()
        max_unique_ratio = 0.6; min_rows_for_ratio_check = 50; max_abs_unique = 100
        rows = len(df)
        if rows > min_rows_for_ratio_check:
            for col in potential_cat_cols:
                 try:
                    unique_count = df[col].nunique()
                    unique_ratio = unique_count / rows
                    if unique_ratio > max_unique_ratio or unique_count > max_abs_unique:
                         if col in categorical_cols: categorical_cols.remove(col)
                 except Exception: # Handle non-hashable types etc.
                      if col in categorical_cols: categorical_cols.remove(col)

        # Ensure no overlap after refinement
        numeric_cols = list(set(numeric_cols) - set(bool_cols) - set(datetime_cols))
        categorical_cols = list(set(categorical_cols) - set(numeric_cols) - set(bool_cols) - set(datetime_cols))
        datetime_cols = list(set(datetime_cols) - set(numeric_cols) - set(bool_cols))
        bool_cols = list(set(bool_cols)) # Ensure it's a list

        # Combine bools into categoricals for some wizard functions maybe? Or handle bool separately
        # Let's return bools separately for now
        return numeric_cols, categorical_cols, datetime_cols, bool_cols

    except Exception as e:
        st.warning(f"Could not determine column types accurately: {e}")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        other_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        return numeric_cols, other_cols, [], []


def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()
def save_plot_to_bytes(fig):
    img_bytes = io.BytesIO()
    try:
        if fig is not None and isinstance(fig, plt.Figure):
             fig.savefig(img_bytes, format='png', bbox_inches='tight', dpi=150)
             plt.close(fig); img_bytes.seek(0); return img_bytes.getvalue()
        else: return None
    except Exception: plt.close(fig); return None # Ensure close on error

# Caching functions for EDA Reports (keep existing generate_profile_report, generate_sweetviz_report)
@st.cache_data(show_spinner="Generating YData Profile Report (this can take a while)...")
def generate_profile_report(_df, _title="Data Profile Report"):
    if _df is None or _df.empty: st.warning("No data for profile report."); return None
    try:
        profile = ProfileReport(_df, title=_title, minimal=True)
        return profile.to_html()
    except Exception as e: st.error(f"Profile report error: {e}"); st.error(traceback.format_exc()); return None

@st.cache_data(show_spinner="Generating Sweetviz Report (this can take a while)...")
def generate_sweetviz_report(_df):
     if _df is None or _df.empty: st.warning("No data for Sweetviz report."); return None
     try:
         report = sv.analyze(_df); return report.show_html(filepath=None, open_browser=False, layout='vertical', scale=None)
     except Exception as e: st.error(f"Sweetviz report error: {e}"); st.error(traceback.format_exc()); return None


# --- Initialize Session State ---
st.session_state.setdefault('data_source', None)
st.session_state.setdefault('df', None)
st.session_state.setdefault('db_params', {'server': None, 'database': None, 'table': None})
st.session_state.setdefault('generated_code', "") # Wizard code
st.session_state.setdefault('result_type', None) # Wizard result type
st.session_state.setdefault('uploaded_file_state', None)
st.session_state.setdefault('selected_category', list(ACTION_CATEGORIES.keys())[0]) # Wizard category
st.session_state.setdefault('show_profile_report', False)
st.session_state.setdefault('show_sweetviz_report', False)
st.session_state.setdefault('current_action_result_display', None) # Wizard result object
# State for ML Lite tabs
st.session_state.setdefault('ml_lite_target', None) # Target selected across ML lite tabs


# --- App Layout ---
st.set_page_config(layout="wide", page_title="Data Analysis & ML Insights")
st.title("📊 Data Analysis & ML Insights Wizard 🧙‍♂️")
st.markdown("Connect to data, explore with the Wizard, or gain ML-related insights.")

# --- Sidebar ---
with st.sidebar:
    st.header("🔄 Session Control")
    if st.button("Start New Session / Reset All"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear: del st.session_state[key]
        st.rerun()

    st.divider()
    st.header("🔗 Connect to Data")
    db_connection_available = bool(DB_SERVERS)
    connection_options = ['Upload File (CSV/Excel)']
    if db_connection_available: connection_options.append('Connect to Database')
    current_data_source_index = 0 if st.session_state.data_source == 'Upload File (CSV/Excel)' else 1 if st.session_state.data_source == 'Connect to Database' and db_connection_available else None

    data_source_option = st.radio(
        "Data Source:", connection_options, key='data_source_radio', index=current_data_source_index,
        on_change=lambda: st.session_state.update(df=None, generated_code="", result_type=None, current_action_result_display=None, show_profile_report=False, show_sweetviz_report=False, ml_lite_target=None) # Reset state on source change
    )
    if data_source_option != st.session_state.data_source: st.session_state.data_source = data_source_option; st.rerun()

    # --- File Upload ---
    if st.session_state.data_source == 'Upload File (CSV/Excel)':
        st.subheader("📤 Upload")
        uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx', 'xls'], key='file_uploader', label_visibility="collapsed")
        if uploaded_file is not None:
            file_identifier = (uploaded_file.name, uploaded_file.size, uploaded_file.type)
            if st.session_state.uploaded_file_state != file_identifier:
                st.session_state.uploaded_file_state = file_identifier
                st.session_state.df = None # Clear before loading
                # Reset all results/ML state
                st.session_state.generated_code = "" ; st.session_state.result_type = None ; st.session_state.current_action_result_display = None
                st.session_state.show_profile_report = False ; st.session_state.show_sweetviz_report = False
                st.session_state.ml_lite_target = None
                # Load data
                st.session_state.df = load_file_data(uploaded_file)
                if st.session_state.df is not None: st.success(f"Loaded `{uploaded_file.name}`"); st.rerun()
                else: st.session_state.uploaded_file_state = None

    # --- Database Connection ---
    elif st.session_state.data_source == 'Connect to Database':
        st.subheader("🗄️ Database")
        if not db_connection_available: st.error(f"DB connection disabled ('{DB_CONFIG_FILE}'?).")
        else:
            server_names = list(DB_SERVERS.keys())
            current_server = st.session_state.db_params.get('server')
            server_index = server_names.index(current_server) if current_server in server_names else None
            selected_server = st.selectbox("Server", server_names, index=server_index, placeholder="Choose...", key="db_server_select")
            if selected_server != current_server:
                 st.session_state.db_params = {'server': selected_server, 'database': None, 'table': None} ; st.session_state.df = None; st.rerun()

            selected_db = None
            current_server = st.session_state.db_params.get('server')
            if current_server:
                available_dbs = get_databases(current_server)
                current_db = st.session_state.db_params.get('database')
                db_index = available_dbs.index(current_db) if current_db in available_dbs else None
                if available_dbs:
                    selected_db = st.selectbox("Database", available_dbs, index=db_index, placeholder="Choose...", key="db_select")
                    if selected_db != current_db:
                        st.session_state.db_params['database'] = selected_db; st.session_state.db_params['table'] = None; st.session_state.df = None; st.rerun()
                else: st.session_state.db_params['database'] = None

            selected_table = None
            current_db = st.session_state.db_params.get('database')
            if current_server and current_db:
                available_tables = get_tables(current_server, current_db)
                current_table = st.session_state.db_params.get('table')
                table_index = available_tables.index(current_table) if current_table in available_tables else None
                if available_tables:
                    selected_table = st.selectbox("Table", available_tables, index=table_index, placeholder="Choose...", key="db_table_select")
                    if selected_table != current_table:
                         st.session_state.db_params['table'] = selected_table; st.session_state.df = None; st.rerun()
                else: st.session_state.db_params['table'] = None

            current_table = st.session_state.db_params.get('table')
            if current_table:
                query_method = st.radio("Fetch", ("TOP 1000 Rows", "Custom Query"), key="db_query_method")
                sql_query = ""
                if query_method == "TOP 1000 Rows":
                     table_parts = current_table.split('.'); quoted_table = f"[{table_parts[0]}].[{table_parts[1]}]" if len(table_parts) == 2 else f"[{current_table}]"
                     sql_query = f"SELECT TOP 1000 * FROM {quoted_table};"; st.text_area("SQL:", value=sql_query, height=100, disabled=True, key="db_sql_display")
                else:
                    default_custom_sql = f"SELECT * FROM {current_table};" if current_table else "SELECT * FROM ..."
                    st.session_state.setdefault('custom_sql_input', default_custom_sql)
                    sql_query = st.text_area("SQL Query:", value=st.session_state.custom_sql_input, height=150, key="db_sql_custom", on_change=lambda: st.session_state.update(custom_sql_input=st.session_state.db_sql_custom))

                if st.button("Fetch Data", key="db_fetch_button"):
                    current_sql_to_run = sql_query; current_server_fetch = st.session_state.db_params.get('server'); current_database_fetch = st.session_state.db_params.get('database')
                    if current_sql_to_run and current_server_fetch and current_database_fetch:
                        # Reset state before fetching
                        st.session_state.df = None ; st.session_state.generated_code = "" ; st.session_state.result_type = None
                        st.session_state.show_profile_report = False ; st.session_state.show_sweetviz_report = False
                        st.session_state.current_action_result_display = None ; st.session_state.uploaded_file_state = None
                        st.session_state.ml_lite_target = None
                        # Fetch data
                        fetched_df = fetch_data_from_db(current_server_fetch, current_database_fetch, current_sql_to_run)
                        st.session_state.df = fetched_df
                        if fetched_df is not None and not fetched_df.empty: st.success(f"Fetched data."); st.rerun()
                        elif fetched_df is not None: st.warning("Query returned no data."); st.rerun()
                    else: st.warning("Ensure Server, DB, Table/Query.")

# --- Main Panel ---
if st.session_state.df is not None:
    st.header("Preview of Loaded Data")
    df_display = st.session_state.df.copy()
    try: # Convert bools to string for st.dataframe display
        bool_cols = df_display.select_dtypes(include=['boolean', 'bool']).columns
        for col in bool_cols: df_display[col] = df_display[col].astype(str)
    except Exception: pass # Ignore errors during display conversion
    st.dataframe(df_display, height=300, use_container_width=True)

    # --- Tabs for different functionalities ---
    tab_titles = ["Automated EDA", "Data Analysis Wizard", "ML Readiness", "XAI Insights", "Model Advisor"]
    tab_eda, tab_wizard, tab_readiness, tab_xai, tab_advisor = st.tabs(tab_titles)

    # --- Tab: Automated EDA ---
    with tab_eda:
        st.subheader("📊 Automated EDA Reports")
        st.markdown("Generate comprehensive reports for quick data understanding.")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
             if st.button("Generate YData Profile Report", key="profile_report_btn", use_container_width=True):
                 st.session_state.show_profile_report = True; st.session_state.show_sweetviz_report = False
                 st.session_state.current_action_result_display = None; st.rerun()
        with report_col2:
            if st.button("Generate Sweetviz Report", key="sweetviz_report_btn", use_container_width=True):
                st.session_state.show_sweetviz_report = True; st.session_state.show_profile_report = False
                st.session_state.current_action_result_display = None; st.rerun()

        report_placeholder = st.container()
        with report_placeholder: # Display reports conditionally
            if st.session_state.get('show_profile_report', False):
                report_title = f"Profile Report {st.session_state.uploaded_file_state[0]}" if st.session_state.uploaded_file_state else "Profile Report"
                report_html = generate_profile_report(st.session_state.df, report_title)
                if report_html: with st.expander("YData Report", expanded=True): components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate YData Report.")
            if st.session_state.get('show_sweetviz_report', False):
                report_html = generate_sweetviz_report(st.session_state.df)
                if report_html: with st.expander("Sweetviz Report", expanded=True): components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate Sweetviz Report.")

    # --- Tab: Data Analysis Wizard ---
    with tab_wizard:
        st.header("🧙‍♂️ Data Analysis Wizard")
        st.markdown("Perform common data tasks. Results and generated code appear below.")

        df_wizard = st.session_state.df
        if df_wizard is None or df_wizard.empty: st.warning("No data loaded.")
        else:
            numeric_cols_wiz, categorical_cols_wiz, datetime_cols_wiz, bool_cols_wiz = get_column_types(df_wizard)
            all_cols_wiz = df_wizard.columns.tolist()

            st.subheader("1. Select Action Category")
            selected_category_index = list(ACTION_CATEGORIES.keys()).index(st.session_state.selected_category)
            selected_category = st.radio("Category:", list(ACTION_CATEGORIES.keys()), index=selected_category_index, horizontal=True, key="category_radio_wizard")
            if selected_category != st.session_state.selected_category: st.session_state.selected_category = selected_category; st.rerun()

            st.subheader("2. Choose Specific Action & Configure")
            action_options = ACTION_CATEGORIES[selected_category]
            st.session_state.setdefault(f'selected_action_{selected_category}', action_options[0])
            selected_action_key = f'selected_action_{selected_category}'
            current_action_in_state = st.session_state.get(selected_action_key, action_options[0])
            selected_action_index = action_options.index(current_action_in_state) if current_action_in_state in action_options else 0
            action = st.selectbox(f"Action in '{selected_category}':", action_options, index=selected_action_index, key=f"action_select_{selected_category}")
            st.session_state[selected_action_key] = action

            # --- Wizard Action Implementation Block ---
            code_string = "# Select action and options"
            action_configured = False

            try: # Wrap configuration UI
                # === Basic Info ===
                if action == "Show Shape": code_string = "result_data = df.shape"; action_configured = True
                elif action == "Show Columns & Types":
                    code_string = "import io\nbuffer = io.StringIO()\ndf.info(buf=buffer)\nresult_data = buffer.getvalue()"; action_configured = True
                elif action == "Show Basic Statistics (Describe)":
                    sel_cols_desc = st.multiselect("Columns (Optional):", all_cols_wiz, default=numeric_cols_wiz + datetime_cols_wiz + bool_cols_wiz, key=f"desc_{action}")
                    code_string = f"result_data = df[{sel_cols_desc}].describe(include='all')" if sel_cols_desc else "result_data = df.describe(include='all')"; action_configured = True
                elif action == "Show Missing Values":
                    code_string = "miss_counts = df.isnull().sum()\nresult_data = miss_counts[miss_counts > 0].reset_index().rename(columns={'index':'Col', 0:'Missing'}).sort_values('Missing', ascending=False)"; action_configured = True
                elif action == "Show Unique Values (Categorical)":
                    cat_col_u = st.selectbox("Categorical Column:", categorical_cols_wiz, key=f"unique_{action}")
                    if cat_col_u: max_u_show = 100; code_string = f"u_vals=df['{cat_col_u}'].unique(); n_u=len(u_vals)\nif n_u>{max_u_show}: result_data=f'{{n_u}} unique (showing {max_u_show}):\\n' + '\\n'.join(map(str,u_vals[:{max_u_show}]))\nelse: result_data=pd.DataFrame(u_vals, columns=['Unique']).dropna()"; action_configured = True
                elif action == "Count Values (Categorical)":
                    cat_col_c = st.selectbox("Categorical Column:", categorical_cols_wiz, key=f"count_{action}")
                    if cat_col_c: norm = st.checkbox("%?", key=f"count_norm_{action}"); code_string = f"cts=df['{cat_col_c}'].value_counts(normalize={norm}, dropna=False).reset_index()\ncts.columns=['{cat_col_c}', ('%' if {norm} else '#')]; result_data=cts"; action_configured = True
                # === Data Cleaning ===
                elif action == "Drop Columns": drop_c = st.multiselect("Drop:", all_cols_wiz, key=f"drop_{action}"); if drop_c: code_string = f"result_data = df.drop(columns={drop_c})"; action_configured = True
                elif action == "Rename Columns":
                    map_r = {}; opts_r = sorted(all_cols_wiz); cols_r = st.multiselect("Rename:", opts_r, key=f"r_sel_{action}")
                    for c in cols_r: k=f"r_in_{''.join(ch if ch.isalnum() else '_' for ch in c)}_{action}"; n=st.text_input(f"'{c}' ->", c, key=k); map_r[c]=n if n!=c and n and n.isidentifier() else None
                    if cols_r and all(map_r.get(c) for c in cols_r): code_string = f"result_data = df.rename(columns={{c:n for c,n in {map_r}.items() if n}})"; action_configured=True
                    elif cols_r: st.caption("Enter valid new names.")
                elif action == "Handle Missing Data":
                    cols_fna=st.multiselect("Columns:", all_cols_wiz, key=f"fna_c_{action}")
                    if cols_fna:
                        m=st.radio("Method:", ["Value", "Mean", "Median", "Mode", "ffill", "bfill", "Drop Rows"], key=f"fna_m_{action}")
                        lns=["r=df.copy()"]; vop=True
                        if m=="Value": v=st.text_input("Value:","0",key=f"fna_v_{action}"); p=repr(float(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else v); [lns.append(f"r['{c}']=r['{c}'].fillna({p})") for c in cols_fna]
                        elif m=="Drop Rows": lns.append(f"r=r.dropna(subset={cols_fna})")
                        else:
                            for c in cols_fna:
                                if m in ["Mean","Median"] and c not in numeric_cols_wiz: st.warning(f"{m} needs numeric '{c}'"); vop=False; break
                                elif m=="Mean": lns.append(f"r['{c}']=r['{c}'].fillna(pd.to_numeric(r['{c}'],errors='coerce').mean())")
                                elif m=="Median": lns.append(f"r['{c}']=r['{c}'].fillna(pd.to_numeric(r['{c}'],errors='coerce').median())")
                                elif m=="Mode": lns.append(f"md=r['{c}'].mode(); if not md.empty: r['{c}']=r['{c}'].fillna(md[0])")
                                elif m=="ffill": lns.append(f"r['{c}']=r['{c}'].ffill()")
                                elif m=="bfill": lns.append(f"r['{c}']=r['{c}'].bfill()")
                        if vop: code_string = "\n".join(lns)+"\nresult_data=r"; action_configured=True
                elif action == "Drop Duplicate Rows": sub=st.multiselect("Cols (Optional):", all_cols_wiz, key=f"dd_s_{action}"); kp=st.radio("Keep:",['first','last',False],format_func=lambda x:str(x) if isinstance(x,str) else "Drop All", key=f"dd_k_{action}"); code_string=f"result_data=df.drop_duplicates(subset=({sub} if {sub} else None), keep=('{kp}' if isinstance('{kp}',str) else False))"; action_configured=True
                elif action == "Change Data Type":
                    tc=st.selectbox("Column:", all_cols_wiz, key=f"dt_c_{action}"); tt=st.selectbox("To Type:",['infer','Int64','float','str','datetime','category','boolean'], key=f"dt_t_{action}")
                    if tc and tt:
                        lns=["r=df.copy()"]; dt_inf=True
                        if tt=='infer': lns.append(f"r['{tc}']=r['{tc}'].convert_dtypes()")
                        elif tt=='datetime': dt_inf=st.checkbox("Infer Format?",True,key=f"dt_inf_{action}"); f_str=f",infer_datetime_format={dt_inf}" if dt_inf else ""; lns.append(f"r['{tc}']=pd.to_datetime(r['{tc}'],errors='coerce'{f_str})")
                        elif tt in ['Int64','float']: lns.append(f"r['{tc}']=pd.to_numeric(r['{tc}'],errors='coerce')"); lns.append(f"if '{tt}'=='Int64': r['{tc}']=r['{tc}'].astype('Int64')")
                        elif tt=='boolean': mp=repr({'true':True,'false':False,'1':True,'0':False,'yes':True,'no':False,'t':True,'f':False,'':pd.NA,'nan':pd.NA,'none':pd.NA,'null':pd.NA}); lns.append(f"bm={{str(k).lower():v for k,v in {mp}.items()}}; r['{tc}']=r['{tc}'].astype(str).str.lower().map(bm).astype('boolean')")
                        else: lns.append(f"r['{tc}']=r['{tc}'].astype('{tt}')")
                        code_string="\n".join(lns)+"\nresult_data=r"; action_configured=True
                elif action == "String Manipulation (Trim, Case, Replace)":
                    sc=st.selectbox("Column:", all_cols_wiz, key=f"sm_c_{action}");
                    if sc:
                        so=st.radio("Op:",["Trim","Upper","Lower","Title","Replace"], key=f"sm_o_{action}"); lns=["r=df.copy()"]; bsc=f"r['{sc}'].astype(str)"
                        if so=="Trim": lns.append(f"r['{sc}']={bsc}.str.strip()")
                        elif so=="Upper": lns.append(f"r['{sc}']={bsc}.str.upper()")
                        elif so=="Lower": lns.append(f"r['{sc}']={bsc}.str.lower()")
                        elif so=="Title": lns.append(f"r['{sc}']={bsc}.str.title()")
                        elif so=="Replace": fi=st.text_input("Find:",key=f"sm_f_{action}"); rp=st.text_input("Replace:",key=f"sm_r_{action}"); rg=st.checkbox("Regex?",key=f"sm_rg_{action}"); lns.append(f"r['{sc}']={bsc}.str.replace({repr(fi)},{repr(rp)},regex={rg})"); action_configured=True
                        else: action_configured=True
                        if action_configured: code_string="\n".join(lns)+"\nresult_data=r"
                elif action == "Extract from Text (Regex)":
                    xc=st.selectbox("Column:",all_cols_wiz,key=f"ex_c_{action}"); pt=st.text_input("Pattern:",placeholder=r"(\d+)",key=f"ex_p_{action}"); ns=st.text_input("New Names (CSV):",placeholder="num1,num2",key=f"ex_n_{action}")
                    if xc and pt:
                        lns=["r=df.copy()"]; xcs=f"r['{xc}'].astype(str)"; nl=[n.strip() for n in ns.split(',') if n.strip()] if ns else None
                        if nl: lns.append(f"xt= {xcs}.str.extract(r{repr(pt)}); if xt.shape[1]==len({repr(nl)}): r[{repr(nl)}]=xt"); lns.append(f"else: print(f'Warn: Extracted {{xt.shape[1]}} groups, needed {{len({repr(nl)})}}')")
                        else: lns.append(f"xt_data={xcs}.str.extract(r{repr(pt)}); result_data=xt_data") # Assign extracted to result if no names
                        code_string="\n".join(lns)+("\nresult_data=r" if nl else ""); action_configured=True
                elif action == "Date Component Extraction (Year, Month...)":
                    dc=st.selectbox("Column:",all_cols_wiz,key=f"dc_c_{action}"); cp=st.selectbox("Component:",["Year","Month","Day","Hour","Minute","Second","DayOfWeek","DayName","MonthName","Quarter","WeekOfYear"],key=f"dc_p_{action}")
                    if dc and cp:
                        cm={"Year":".dt.year","Month":".dt.month","Day":".dt.day","Hour":".dt.hour","Minute":".dt.minute","Second":".dt.second","DayOfWeek":".dt.dayofweek","DayName":".dt.day_name()","MonthName":".dt.month_name()","Quarter":".dt.quarter","WeekOfYear":".dt.isocalendar().week"}
                        nn=f"{dc}_{cp.lower()}"; lns=[f"r=df.copy()", f"td=pd.to_datetime(r['{dc}'],errors='coerce')", f"r['{nn}']=np.nan", f"vd=td.notna()", f"if vd.any():", f"    ec=td[vd]{cm[cp]}", f"    r.loc[vd,'{nn}']=ec", f"    if pd.api.types.is_integer_dtype(ec): r['{nn}']=r['{nn}'].astype('Int64')"]
                        code_string="\n".join(lns)+"\nresult_data=r"; action_configured=True
                # === Data Transformation ===
                elif action == "Filter Data":
                    fc=st.selectbox("Filter Column:",all_cols_wiz,key=f"f_c_{action}")
                    if fc:
                        dt=df_wizard[fc].dtype; isn=pd.api.types.is_numeric_dtype(dt); isd=pd.api.types.is_datetime64_any_dtype(dt); isb=pd.api.types.is_bool_dtype(dt) or pd.api.types.is_bool(dt)
                        ops_n=['==','!=','>','<','>=','<=','missing','not missing']; ops_s=['==','!=','contains','!contains','startswith','endswith','in list','!in list','missing','not missing']; ops_d=ops_n; ops_b=['is True','is False','missing','not missing']
                        op=None; val=None; lns=[]
                        if isn: op=st.selectbox("Op:",ops_n,key=f"f_on_{action}");
                        elif isd: op=st.selectbox("Op:",ops_d,key=f"f_od_{action}"); lns.append(f"s=pd.to_datetime(df['{fc}'],errors='coerce')")
                        elif isb: op=st.selectbox("Op:",ops_b,key=f"f_ob_{action}"); lns.append(f"s=df['{fc}'].astype('boolean')")
                        else: op=st.selectbox("Op:",ops_s,key=f"f_os_{action}"); lns.append(f"s=df['{fc}'].astype(str)")
                        # Get value based on type and op
                        if op not in ['missing','not missing']:
                            if isn: val=st.number_input(f"Value:",value=0.0,format="%g",key=f"f_vn_{action}"); lns.append(f"fs=pd.to_numeric(df['{fc}'],errors='coerce'){op}{val}"); action_configured=True
                            elif isd: vd=st.date_input(f"Date:",key=f"f_vd_{action}"); val=pd.Timestamp(f"{vd}"); lns.append(f"fs=s{op}pd.Timestamp('{val}')"); action_configured=True
                            elif isb: action_configured=True # No value needed for is True/False
                            else: # String ops
                                vs=st.text_input(f"Value(s):",key=f"f_vs_{action}")
                                if vs is not None: # Can be empty string
                                    if op=='contains': lns.append(f"fs=s.str.contains({repr(vs)},case=False,na=False)")
                                    elif op=='!contains': lns.append(f"fs=~s.str.contains({repr(vs)},case=False,na=False)")
                                    elif op=='startswith': lns.append(f"fs=s.str.startswith({repr(vs)},na=False)")
                                    elif op=='endswith': lns.append(f"fs=s.str.endswith({repr(vs)},na=False)")
                                    elif op=='in list': lv=[v.strip() for v in vs.split(',') if v.strip()]; lns.append(f"fs=s.isin({lv})") if lv else st.caption("Enter list"); action_configured=bool(lv)
                                    elif op=='!in list': lv=[v.strip() for v in vs.split(',') if v.strip()]; lns.append(f"fs=~s.isin({lv})"); action_configured=True # Works with empty list
                                    else: lns.append(f"fs=s{op}{repr(vs)}")
                                    if action_configured is None : action_configured=True # Default to true if value entered
                        # Handle missing/not missing and boolean True/False
                        elif op=='missing': lns.append(f"fs=df['{fc}'].isnull()" + ("| (s=='')" if not (isn or isd or isb) else "")); action_configured=True
                        elif op=='not missing': lns.append(f"fs=df['{fc}'].notnull()" + ("& (s!='')" if not (isn or isd or isb) else "")); action_configured=True
                        elif op=='is True': lns.append(f"fs=s==True"); action_configured=True
                        elif op=='is False': lns.append(f"fs=s==False"); action_configured=True
                        # Finalize code string
                        if action_configured: lns.append("result_data=df[fs]"); code_string="\n".join(lns)

                elif action == "Sort Data":
                    sc=st.multiselect("Sort By:",all_cols_wiz,key=f"s_c_{action}")
                    if sc: so=[st.radio(f"Order '{c}':",["Asc","Desc"],key=f"s_o_{c}",horizontal=True)=="Asc" for c in sc]; code_string=f"result_data=df.sort_values(by={sc},ascending={so})"; action_configured=True
                elif action == "Select Columns":
                     slc=st.multiselect("Keep:",all_cols_wiz,default=all_cols_wiz,key=f"sl_c_{action}");
                     if slc: code_string=f"result_data=df[{slc}]"; action_configured=True
                     else: st.warning("Select columns.")
                elif action == "Create Calculated Column (Basic Arithmetic)":
                     nn=st.text_input("New Name:",key=f"cc_n_{action}"); c1=st.selectbox("Col1/Const:",[None]+numeric_cols_wiz,key=f"cc_c1_{action}"); op=st.selectbox("Op:",['+','-','*','/'],key=f"cc_o_{action}"); c2=st.selectbox("Col2/Const:",[None]+numeric_cols_wiz,key=f"cc_c2_{action}"); cnst=st.text_input("Const Val:","0",key=f"cc_cn_{action}")
                     if nn and nn.isidentifier() and op and (c1 or c2 or cnst):
                         try: cv=float(cnst); t1=f"pd.to_numeric(df['{c1}'],errors='coerce')" if c1 else str(cv); t2=f"pd.to_numeric(df['{c2}'],errors='coerce')" if c2 else str(cv)
                         except ValueError: st.error("Invalid const."); continue
                         if c1 or c2:
                             if op=='/' and not c2 and cv==0: st.error("Div by zero const.")
                             else: code_string=f"r=df.copy(); r['{nn}']={t1}{op}{t2}; result_data=r"; action_configured=True
                         else: st.warning("Select col.")
                     elif nn and not nn.isidentifier(): st.warning("Invalid name.")
                elif action == "Bin Numeric Data (Cut)":
                    bc=st.selectbox("Column:",numeric_cols_wiz,key=f"b_c_{action}");
                    if bc:
                        bm=st.radio("Method:",["Equal Width","Quantiles","Custom"],key=f"b_m_{action}"); nn=st.text_input("New Name:",f"{bc}_bin",key=f"b_n_{action}")
                        if nn and nn.isidentifier():
                            lns=[f"r=df.copy()",f"nc=pd.to_numeric(r['{bc}'],errors='coerce')"]; lbl=st.radio("Labels:",["Numeric","Range"],key=f"b_l_{action}",horizontal=True)=="Numeric"; lp="labels=False" if lbl else ""
                            if bm=="Equal Width": nb=st.slider("Bins:",2,50,5,key=f"b_nb_{action}"); lns.append(f"r['{nn}']=pd.cut(nc,bins={nb},{lp},include_lowest=True,duplicates='drop')"); action_configured=True
                            elif bm=="Quantiles": nq=st.slider("Quantiles:",2,10,4,key=f"b_nq_{action}"); lns.append(f"r['{nn}']=pd.qcut(nc,q={nq},{lp},duplicates='drop')"); action_configured=True
                            elif bm=="Custom": es=st.text_input("Edges (CSV):",placeholder="0,10,100",key=f"b_e_{action}");
                            try: ed=[float(e.strip()) for e in es.split(',') if e.strip()];
                            except ValueError: st.error("Invalid edges."); continue
                            if len(ed)>1: lns.append(f"be={sorted(ed)}; r['{nn}']=pd.cut(nc,bins=be,{lp},include_lowest=True,duplicates='drop')"); action_configured=True
                            elif es: st.warning("Need >=2 edges.")
                            if action_configured: code_string="\n".join(lns)+"\nresult_data=r"
                        elif nn and not nn.isidentifier(): st.warning("Invalid name.")
                elif action == "One-Hot Encode Categorical Column":
                     oc=st.selectbox("Column:",categorical_cols_wiz,key=f"ohe_c_{action}")
                     if oc: df=st.checkbox("Drop First?",False,key=f"ohe_d_{action}"); code_string=f"result_data=pd.get_dummies(df,columns=['{oc}'],prefix='{oc}',drop_first={df},dummy_na=False)"; action_configured=True
                elif action == "Pivot (Simple)":
                     pi=st.selectbox("Index:",all_cols_wiz,key=f"p_i_{action}"); pc=st.selectbox("Columns:",all_cols_wiz,key=f"p_c_{action}"); pv=st.selectbox("Values:",numeric_cols_wiz,key=f"p_v_{action}"); pa=st.selectbox("Agg:",['mean','sum','count','median'],key=f"p_a_{action}")
                     if pi and pc and pv and pa:
                         if pi==pc: st.warning("Index!=Columns")
                         else: code_string=f"p=df.copy(); p['{pv}']=pd.to_numeric(p['{pv}'],errors='coerce'); result_data=pd.pivot_table(p,index='{pi}',columns='{pc}',values='{pv}',aggfunc='{pa}').reset_index()"; action_configured=True
                elif action == "Melt (Unpivot)":
                     ids=st.multiselect("ID Vars:",all_cols_wiz,key=f"m_i_{action}"); dv=[c for c in all_cols_wiz if c not in ids]; vvs=st.multiselect("Value Vars:",dv,default=dv,key=f"m_v_{action}"); vn=st.text_input("Var Name:","Var",key=f"m_vn_{action}"); vln=st.text_input("Value Name:","Val",key=f"m_vln_{action}")
                     if ids and vn and vln and vn.isidentifier() and vln.isidentifier(): vp=f"value_vars={vvs}" if vvs else ""; code_string=f"result_data=pd.melt(df,id_vars={ids},{vp},var_name='{vn}',value_name='{vln}')"; action_configured=True
                     elif not ids: st.warning("Select ID Vars.")
                     elif not (vn and vln and vn.isidentifier() and vln.isidentifier()): st.warning("Valid names.")
                # === Aggregation & Analysis ===
                elif action == "Calculate Single Aggregation (Sum, Mean...)":
                    ac=st.selectbox("Column:",numeric_cols_wiz,key=f"sa_c_{action}"); af=st.selectbox("Function:",['sum','mean','median','min','max','count','nunique','std','var'],key=f"sa_f_{action}")
                    if ac and af: code_string=f"an=pd.to_numeric(df['{ac}'],errors='coerce'); result_data=an.{af}()"; action_configured=True
                elif action == "Group By & Aggregate":
                    gc=st.multiselect("Group By:",categorical_cols_wiz+datetime_cols_wiz,key=f"ga_g_{action}"); na=st.number_input("# Aggs:",1,value=1,key=f"ga_n_{action}"); nl=[]; vc=True; cnc=set()
                    for i in range(na): st.markdown(f"**Agg #{i+1}**"); acg=st.selectbox(f"Column:",all_cols_wiz,key=f"ga_ac_{i}"); afg=st.selectbox(f"Func:",['sum','mean','median','min','max','count','nunique','std','var','first','last'],key=f"ga_af_{i}"); dn=f"{acg}_{afg}".replace('[^A-Za-z0-9_]+','') if acg else f"agg_{i+1}"; nn=st.text_input(f"Name:",dn,key=f"ga_nn_{i}")
                    if not acg: st.warning(f"#{i+1}: Select col."); vc=False
                    elif not nn or not nn.isidentifier(): st.warning(f"#{i+1}: Valid name."); vc=False
                    else: if afg in ['sum','mean','median','std','var']: cnc.add(acg); nl.append(f"{nn}=pd.NamedAgg(column='{acg}',aggfunc='{afg}')")
                    if gc and nl and vc: ncc="\n".join([f"da['{c}']=pd.to_numeric(da['{c}'],errors='coerce')" for c in cnc]); nas=", ".join(nl); code_string=f"da=df.copy(); {ncc}\nresult_data=da.groupby({gc}).agg({nas}).reset_index()"; action_configured=True
                    elif not gc: st.warning("Select Group By.")
                elif action == "Calculate Rolling Window Statistics":
                    rc=st.selectbox("Column:",numeric_cols_wiz,key=f"rol_c_{action}");
                    if rc:
                        ws=st.number_input("Window:",2,value=3,key=f"rol_w_{action}"); rf=st.selectbox("Func:",['mean','sum','median','std','min','max','var'],key=f"rol_f_{action}"); cn=st.checkbox("Center?",False,key=f"rol_cn_{action}"); mp=st.number_input("Min Periods:",1,value=None,placeholder="Default",key=f"rol_mp_{action}"); sbc=st.selectbox("Sort by:",[None]+all_cols_wiz,key=f"rol_s_{action}"); nn=st.text_input("New Name:",f"{rc}_rol_{rf}",key=f"rol_nn_{action}")
                        if nn and nn.isidentifier(): lns=["r=df.copy()"]; if sbc: lns.append(f"r=r.sort_values(by='{sbc}')"); lns.append(f"rcn=pd.to_numeric(r['{rc}'],errors='coerce')"); mpp=f",min_periods={mp}" if mp is not None else ""; lns.append(f"r['{nn}']=rcn.rolling(window={ws},center={cn}{mpp}).{rf}()"); if sbc: lns.append(f"# Sorted by {sbc}"); code_string="\n".join(lns)+"\nresult_data=r"; action_configured=True
                        elif nn and not nn.isidentifier(): st.warning("Invalid name.")
                elif action == "Calculate Cumulative Statistics":
                     cc=st.selectbox("Column:",numeric_cols_wiz,key=f"cum_c_{action}");
                     if cc:
                         cf=st.selectbox("Func:",['sum','prod','min','max'],key=f"cum_f_{action}"); scc=st.selectbox("Sort by:",[None]+all_cols_wiz,key=f"cum_s_{action}"); nn=st.text_input("New Name:",f"{cc}_cum_{cf}",key=f"cum_nn_{action}")
                         if nn and nn.isidentifier(): lns=["r=df.copy()"]; if scc: lns.append(f"r=r.sort_values(by='{scc}')"); lns.append(f"ccn=pd.to_numeric(r['{cc}'],errors='coerce')"); lns.append(f"r['{nn}']=ccn.cum{cf}()"); if scc: lns.append(f"# Sorted by {scc}"); code_string="\n".join(lns)+"\nresult_data=r"; action_configured=True
                         elif nn and not nn.isidentifier(): st.warning("Invalid name.")
                elif action == "Rank Data":
                     rkc=st.selectbox("Column:",numeric_cols_wiz,key=f"rnk_c_{action}");
                     if rkc:
                         ra=st.radio("Order:",["Asc","Desc"],key=f"rnk_o_{action}",horizontal=True)=="Asc"; rm=st.selectbox("Ties:",['average','min','max','first','dense'],key=f"rnk_m_{action}"); nn=st.text_input("New Name:",f"{rkc}_rank",key=f"rnk_nn_{action}")
                         if nn and nn.isidentifier(): code_string=f"r=df.copy(); rcn=pd.to_numeric(r['{rkc}'],errors='coerce'); r['{nn}']=rcn.rank(method='{rm}',ascending={ra},na_option='bottom'); result_data=r"; action_configured=True
                         elif nn and not nn.isidentifier(): st.warning("Invalid name.")
                # === Visualization ===
                elif action == "Plot Histogram (Numeric)":
                    hc=st.selectbox("Column:",numeric_cols_wiz,key=f"h_c_{action}")
                    if hc: bn=st.slider("Bins:",5,100,20,key=f"h_b_{action}"); kd=st.checkbox("KDE?",True,key=f"h_k_{action}"); code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); pcn=pd.to_numeric(df['{hc}'],errors='coerce'); sns.histplot(x=pcn.dropna(),bins={bn},kde={kd},ax=ax); ax.set_title('Hist {hc}'); plt.tight_layout(); result_data=fig"; action_configured=True
                elif action == "Plot Density Plot (Numeric)":
                    dc=st.selectbox("Column:",numeric_cols_wiz,key=f"d_c_{action}")
                    if dc: sh=st.checkbox("Shade?",True,key=f"d_s_{action}"); code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); pcn=pd.to_numeric(df['{dc}'],errors='coerce'); sns.kdeplot(x=pcn.dropna(),fill={sh},ax=ax); ax.set_title('Density {dc}'); plt.tight_layout(); result_data=fig"; action_configured=True
                elif action == "Plot Count Plot (Categorical)":
                    cc=st.selectbox("Column:",categorical_cols_wiz,key=f"cp_c_{action}")
                    if cc: topN=st.checkbox("Top N?",True,key=f"cp_t_{action}"); n=20; if topN: n=st.slider("N:",5,50,20,key=f"cp_n_{action}"); pcode=f"order=pcd['{cc}'].value_counts().nlargest({n}).index"; code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); c='{cc}'; if c in df.columns: pcd=df[[c]].dropna(); if not pcd.empty: po=({pcode} if {topN} and pcd[c].nunique()>{n} else f'order=pcd[c].value_counts().index'); sns.countplot(y=c,data=pcd,{po},ax=ax,color='skyblue'); ax.set_title(f'Counts {{c}}'); plt.tight_layout(); result_data=fig\nelse: result_data=None"; action_configured=True
                elif action == "Plot Bar Chart (Aggregated)":
                     st.info("Use on suitable DF (e.g., Group By result).")
                     pr=st.session_state.get('current_action_result_display'); dfo={"Orig DF":df_wizard}; if isinstance(pr,pd.DataFrame):dfo["Prev Result"]=pr;
                     sk=st.radio("Data:",list(dfo.keys()),horizontal=True,key=f"b_src_{action}"); bdf=dfo[sk];
                     if isinstance(bdf,pd.DataFrame) and len(bdf.columns)>=2:
                         n_b,c_b,_b,_bl=get_column_types(bdf); xb=st.selectbox("X-axis:",c_b,key=f"b_x_{action}"); yb=st.selectbox("Y-axis:",n_b,key=f"b_y_{action}");
                         if xb and yb: code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); psrc['{yb}']=pd.to_numeric(psrc['{yb}'],errors='coerce'); bpd=psrc.nlargest(30,'{yb}'); sns.barplot(x='{xb}',y='{yb}',data=bpd,ax=ax,color='lightcoral'); ax.set_title('Bar: {yb} by {xb}'); plt.xticks(rotation=60,ha='right'); plt.tight_layout(); result_data=fig"; action_configured=True # Pass psrc=bdf later
                         elif not (xb and yb): st.caption("Select axes.")
                     else: st.warning("Data not suitable.")
                elif action == "Plot Line Chart":
                     xl=st.selectbox("X-axis:",all_cols_wiz,key=f"l_x_{action}"); yl=st.selectbox("Y-axis:",numeric_cols_wiz,key=f"l_y_{action}"); hl=st.selectbox("Group By:",[None]+categorical_cols_wiz,key=f"l_h_{action}")
                     if xl and yl: hp=f",hue='{hl}'" if hl else ""; code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); pld=df.copy(); pld['{yl}']=pd.to_numeric(pld['{yl}'],errors='coerce'); xc='{xl}'; try: pld[xc]=pd.to_datetime(pld[xc],errors='coerce') if pd.api.types.is_datetime64_any_dtype(pld[xc]) or 'date' in xc.lower() else pld[xc]; pld.dropna(subset=[xc,'{yl}'],inplace=True); pld=pld.sort_values(by=xc) if not pld.empty else pld; except Exception:pass;\nif not pld.empty: sns.lineplot(x=xc,y='{yl}'{hp},data=pld,ax=ax,marker='o',markersize=4); ax.set_title('Line: {yl} over {xl}'); plt.xticks(rotation=45,ha='right'); plt.tight_layout(); result_data=fig\nelse: result_data=None"; action_configured=True
                elif action == "Plot Scatter Plot":
                     xs=st.selectbox("X-axis:",numeric_cols_wiz,key=f"s_x_{action}"); ys=st.selectbox("Y-axis:",numeric_cols_wiz,key=f"s_y_{action}"); hs=st.selectbox("Color By:",[None]+categorical_cols_wiz,key=f"s_h_{action}"); ss=st.selectbox("Size By:",[None]+numeric_cols_wiz,key=f"s_s_{action}")
                     if xs and ys: hp=f",hue='{hs}'" if hs else ""; sp=f",size='{ss}'" if ss else ""; code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); sd=df.copy(); sd['{xs}']=pd.to_numeric(sd['{xs}'],errors='coerce'); sd['{ys}']=pd.to_numeric(sd['{ys}'],errors='coerce'); if '{ss}'!='None': sd['{ss}']=pd.to_numeric(sd['{ss}'],errors='coerce'); sd.dropna(subset=['{xs}','{ys}'{(', "'+ss+'"') if ss else ''}],inplace=True);\nsns.scatterplot(x='{xs}',y='{ys}'{hp}{sp},data=sd,ax=ax,alpha=0.6); ax.set_title('Scatter: {ys} vs {xs}'); plt.tight_layout(); result_data=fig"; action_configured=True
                elif action == "Plot Box Plot (Numeric vs Cat)":
                     xb=st.selectbox("X-axis (Cat):",categorical_cols_wiz,key=f"b_x_{action}"); yb=st.selectbox("Y-axis (Num):",numeric_cols_wiz,key=f"b_y_{action}")
                     if xb and yb: lcb=st.checkbox("Limit Cats?",True,key=f"b_l_{action}"); tn=15; if lcb: tn=st.slider("Max Cats:",5,50,15,key=f"b_tn_{action}"); code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\nfig,ax=plt.subplots(); lc={lcb}; tn={tn}; xc='{xb}'; yc='{yb}'; pbd=df.copy(); pbd[yc]=pd.to_numeric(pbd[yc],errors='coerce'); pbd.dropna(subset=[xc,yc],inplace=True);\nif pbd.empty: result_data=None\nelse:\n    if lc and pbd[xc].nunique()>tn: pob=pbd[xc].value_counts().nlargest(tn).index; pbd=pbd[pbd[xc].isin(pob)]; sns.boxplot(x=xc,y=yc,data=pbd,order=pob,ax=ax,showfliers=False)\n    else: pob=pbd[xc].value_counts().index; sns.boxplot(x=xc,y=yc,data=pbd,order=pob,ax=ax,showfliers=False)\n    ax.set_title(f'Box: {{yc}} by {{xc}}'); plt.xticks(rotation=60,ha='right'); plt.tight_layout(); result_data=fig"; action_configured=True
                elif action == "Plot Correlation Heatmap (Numeric)":
                     dcc=numeric_cols_wiz[:min(len(numeric_cols_wiz),15)]; cc=st.multiselect("Columns:",numeric_cols_wiz,default=dcc,key=f"ch_c_{action}");
                     if len(cc)>=2: cm=st.selectbox("Method:",['pearson','kendall','spearman'],key=f"ch_m_{action}"); code_string=f"import matplotlib.pyplot as plt\nimport seaborn as sns\ncdf=df[{cc}].copy(); [cdf[c] for c in {cc} if pd.api.types.is_numeric_dtype(cdf[c]) or cdf[c].astype(str).str.match(r'^-?\\d+(\\.\\d+)?$').all()]; [cdf[c] for c in {cc} if cdf[c].isnull().sum() < len(cdf)]; cdf = cdf[[c for c in {cc} if c in cdf.columns and pd.api.types.is_numeric_dtype(cdf[c])]] ; for c in cdf.columns: cdf[c]=pd.to_numeric(cdf[c],errors='coerce'); cmat=cdf.corr(method='{cm}');\nfig,ax=plt.subplots(figsize=(max(6,len(cdf.columns)*0.7),max(5,len(cdf.columns)*0.6))); sns.heatmap(cmat,annot=True,cmap='coolwarm',fmt='.2f',linewidths=.5,ax=ax,annot_kws={{'size':8}}); ax.set_title('Correlation ({cm.capitalize()})'); plt.xticks(rotation=45,ha='right'); plt.yticks(rotation=0); plt.tight_layout(); result_data=fig"; action_configured=True
                     else: st.warning("Need >= 2 columns.")

            except Exception as e: st.error(f"Error configuring '{action}': {e}"); st.error(traceback.format_exc()); code_string="#Config Error"; action_configured=False


            # --- Trigger Wizard Execution ---
            st.subheader("3. Apply Wizard Action")
            apply_col1_wiz, apply_col2_wiz = st.columns([1, 3])
            with apply_col1_wiz: apply_button_pressed_wiz = st.button(f"Apply: {action}", key=f"apply_{action}", use_container_width=True, disabled=not action_configured)
            if not action_configured and action and not action.startswith("---"): with apply_col2_wiz: st.caption("👈 Configure options.")

            if apply_button_pressed_wiz and action_configured:
                if code_string and code_string != "# Select action and options" and not code_string.startswith("# Config Error"):
                    st.session_state.generated_code = code_string
                    st.session_state.result_type = None; st.session_state.show_profile_report = False; st.session_state.show_sweetviz_report = False
                    st.session_state.current_action_result_display = None; report_placeholder.empty()
                    with st.spinner(f"Applying: {action}..."):
                        local_vars = {'df': df_wizard, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'result_data': None, 'io': io}
                        if action == "Plot Bar Chart (Aggregated)" and 'bar_df_source' in locals() and isinstance(bar_df_source, pd.DataFrame): local_vars['psrc'] = bar_df_source # Pass source df for bar plot

                        try:
                            exec(code_string, {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'io': io}, local_vars)
                            res = local_vars.get('result_data'); current_result_type = None; current_result_to_display = None
                            if isinstance(res, pd.DataFrame): current_result_type = 'dataframe'; current_result_to_display = res
                            elif isinstance(res, pd.Series): current_result_type = 'dataframe'; current_result_to_display = res.to_frame()
                            elif isinstance(res, plt.Figure): current_result_type = 'plot'; current_result_to_display = res
                            elif isinstance(res, str) and action == "Show Columns & Types": current_result_type = 'text_info'; current_result_to_display = res
                            elif res is not None : current_result_type = 'scalar_text'; current_result_to_display = res
                            st.session_state.current_action_result_display = current_result_to_display; st.session_state.result_type = current_result_type
                            if current_result_to_display is not None: st.success("Success!")
                            else: st.warning("Action ran, no result produced.")
                            st.rerun()
                        except Exception as e: st.error(f"Exec Error: {e}"); st.error(traceback.format_exc()); st.session_state.current_action_result_display = None
                        with st.expander("Show Code (Executed or Failed)"): st.code(st.session_state.generated_code, language='python')
                else: with apply_col2_wiz: st.warning("Could not apply. Check config.")


            # --- Display Wizard Results ---
            st.subheader("📊 Wizard Results")
            result_display_placeholder_wiz = st.empty()
            with result_display_placeholder_wiz.container():
                current_result = st.session_state.get('current_action_result_display', None); current_type = st.session_state.get('result_type', None)
                if current_result is not None:
                    try:
                        if current_type == 'dataframe':
                            st.dataframe(current_result); res_df = current_result; c1,c2=st.columns(2)
                            with c1: st.download_button("CSV", convert_df_to_csv(res_df), "wiz_res.csv", "text/csv", use_container_width=True, key="dl_csv")
                            with c2: st.download_button("Excel", convert_df_to_excel(res_df), "wiz_res.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_excel")
                        elif current_type == 'plot':
                            if isinstance(current_result, plt.Figure):
                                 st.pyplot(current_result)
                                 # Save bytes after displaying
                                 plot_bytes = save_plot_to_bytes(current_result)
                                 if plot_bytes: st.download_button("PNG", plot_bytes, "wiz_plot.png", "image/png", use_container_width=True, key="dl_plot")
                                 else: st.warning("Plot download failed.")
                            else: st.error("Invalid plot result.")
                        elif current_type == 'scalar_text':
                             st.write("Result:"); st.write(current_result)
                             try: sd=str(current_result).encode('utf-8'); st.download_button("Text", sd, "wiz_res.txt", "text/plain", use_container_width=True, key="dl_scalar")
                             except Exception as e: st.warning(f"Text DL failed: {e}")
                        elif current_type == 'text_info':
                             st.text(current_result); st.download_button("Info TXT", current_result.encode('utf-8'), "df_info.txt", "text/plain", use_container_width=True, key="dl_info")

                        if st.session_state.generated_code: with st.expander("Show Code"): st.code(st.session_state.generated_code, language='python')
                    except Exception as e: st.error(f"Result Display Error: {e}"); st.error(traceback.format_exc())
                else:
                     if not st.session_state.get('show_profile_report', False) and not st.session_state.get('show_sweetviz_report', False):
                         st.caption("Apply action to see results.")


    # --- Tab: ML Readiness ---
    with tab_readiness:
        st.header("🩺 ML Data Readiness Assessment")
        st.markdown("Analyze data suitability for common ML tasks.")

        df_ready = st.session_state.df
        if df_ready is None or df_ready.empty: st.warning("Load data first.")
        else:
            cols_ready = df_ready.columns.tolist()
            num_cols_r, cat_cols_r, dt_cols_r, bool_cols_r = get_column_types(df_ready)

            # Select Potential Target
            target_options = [None] + cols_ready
            target_r = st.selectbox("Select Potential Target Column:", target_options,
                                    index=target_options.index(st.session_state.ml_lite_target) if st.session_state.ml_lite_target in target_options else 0,
                                    key="ml_ready_target")
            if target_r != st.session_state.ml_lite_target:
                st.session_state.ml_lite_target = target_r
                st.rerun() # Rerun to update analysis based on target

            st.subheader("Analysis Results")

            if target_r:
                st.markdown(f"**Target Variable Analysis (`{target_r}`)**")
                target_series = df_ready[target_r]
                target_dtype = target_series.dtype
                target_missing = target_series.isnull().sum()
                target_missing_pct = (target_missing / len(df_ready)) * 100

                problem_suggestion = "Undetermined"
                if pd.api.types.is_numeric_dtype(target_dtype) and not pd.api.types.is_bool_dtype(target_dtype):
                    # Check if it looks discrete / low cardinality integer - maybe classification?
                    if pd.api.types.is_integer_dtype(target_dtype) and target_series.nunique() < 25:
                         problem_suggestion = f"Likely **Classification** (Integer with {target_series.nunique()} unique values)"
                    else:
                         problem_suggestion = "Likely **Regression** (Numeric)"
                elif pd.api.types.is_bool_dtype(target_dtype) or pd.api.types.is_boolean_dtype(target_dtype):
                     problem_suggestion = f"Likely **Classification** (Boolean - {target_series.nunique()} unique values)"
                elif pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype):
                     problem_suggestion = f"Likely **Classification** (Categorical - {target_series.nunique()} unique values)"

                st.metric("Suggested Problem Type", problem_suggestion)
                st.metric("Missing Target Values", f"{target_missing} ({target_missing_pct:.1f}%)")
                if target_missing_pct > 10: st.warning("High percentage of missing target values. Consider imputation or removal before ML.")

            st.markdown("**Feature Analysis**")
            issues = []
            high_missing_thresh = 20 # %
            high_cardinality_thresh = 50 # unique values
            skewness_thresh = 1.0 # absolute skewness
            near_zero_var_thresh = 0.01 # std dev threshold (relative?) or use nunique=1

            feature_analysis_data = []
            for col in cols_ready:
                 if col == target_r: continue # Skip target
                 series = df_ready[col]
                 dtype = str(series.dtype)
                 missing_pct = (series.isnull().sum() / len(df_ready)) * 100
                 num_unique = series.nunique()
                 analysis = {"Column": col, "DataType": dtype, "Missing (%)": f"{missing_pct:.1f}", "Unique Vals": num_unique}
                 remarks = []

                 if missing_pct > high_missing_thresh: remarks.append(f"High Missing ({missing_pct:.1f}%)")
                 if col in cat_cols_r and num_unique > high_cardinality_thresh: remarks.append(f"High Cardinality ({num_unique})")
                 if col in num_cols_r:
                      try:
                          skew_val = skew(series.dropna())
                          if abs(skew_val) > skewness_thresh: remarks.append(f"Skewed ({skew_val:.2f})")
                      except: pass # Ignore skewness errors
                      try:
                           # NZV check: std dev close to zero OR only 1 unique non-NA value
                           if series.nunique() <= 1: remarks.append("Near Zero Variance (Constant/Single Value)")
                           # std dev check needs care with scale - maybe check if std is tiny fraction of mean?
                           # Or simpler: just check nunique <= 1
                      except: pass
                 analysis["Remarks"] = ", ".join(remarks) if remarks else "OK"
                 feature_analysis_data.append(analysis)

            if feature_analysis_data:
                st.dataframe(pd.DataFrame(feature_analysis_data), use_container_width=True)
            else:
                 st.info("No features to analyze (only target selected?).")

            st.markdown("**Overall Assessment**")
            num_high_missing = sum(1 for item in feature_analysis_data if "High Missing" in item["Remarks"])
            num_high_cardinality = sum(1 for item in feature_analysis_data if "High Cardinality" in item["Remarks"])
            num_skewed = sum(1 for item in feature_analysis_data if "Skewed" in item["Remarks"])
            num_nzv = sum(1 for item in feature_analysis_data if "Near Zero Variance" in item["Remarks"])

            readiness_score = 100
            if target_missing_pct > 30: readiness_score -= 30
            elif target_missing_pct > 10: readiness_score -= 15
            readiness_score -= num_high_missing * 5
            readiness_score -= num_high_cardinality * 3
            readiness_score -= num_skewed * 2
            readiness_score -= num_nzv * 10

            readiness_score = max(0, readiness_score) # Floor at 0

            if readiness_score >= 80: color="green"
            elif readiness_score >= 50: color="orange"
            else: color="red"

            st.markdown(f"#### Readiness Score: <span style='color:{color}; font-size: 1.5em;'>{readiness_score}/100</span>", unsafe_allow_html=True)
            st.caption(f"Summary: Target Missing: {target_missing_pct:.1f}%, High Missing Feats: {num_high_missing}, High Cardinality Feats: {num_high_cardinality}, Skewed Feats: {num_skewed}, NZV Feats: {num_nzv}")
            st.caption("Score is heuristic. Higher scores suggest data is cleaner and more directly suitable for standard ML models. Lower scores indicate more preprocessing may be required.")


    # --- Tab: XAI Insights ---
    with tab_xai:
        st.header("🔍 XAI Feature Insights")
        st.markdown("Explore relationships between features and a potential target variable (without training a model).")

        df_xai = st.session_state.df
        if df_xai is None or df_xai.empty: st.warning("Load data first.")
        else:
            cols_xai = df_xai.columns.tolist()
            num_cols_x, cat_cols_x, dt_cols_x, bool_cols_x = get_column_types(df_xai)

            target_options_xai = [None] + num_cols_x + cat_cols_x + bool_cols_x # Allow most types as target for insights
            target_xai = st.selectbox("Select Target Variable for Insights:", target_options_xai,
                                      index=target_options_xai.index(st.session_state.ml_lite_target) if st.session_state.ml_lite_target in target_options_xai else 0,
                                      key="xai_target")
            if target_xai != st.session_state.ml_lite_target:
                st.session_state.ml_lite_target = target_xai
                st.rerun()

            if target_xai:
                target_series_xai = df_xai[target_xai].copy()
                features_xai = [col for col in cols_xai if col != target_xai]
                numeric_features_xai = [f for f in num_cols_x if f in features_xai]
                categorical_features_xai = [f for f in cat_cols_x if f in features_xai]
                bool_features_xai = [f for f in bool_cols_x if f in features_xai] # Use identified bool cols

                # Determine problem type for plotting/scoring
                is_target_numeric = pd.api.types.is_numeric_dtype(target_series_xai.dtype) and target_series_xai.nunique() > 20 # Heuristic: >20 unique numeric = regression target
                is_target_classification = not is_target_numeric # Assume classification otherwise

                st.subheader("Feature Importance / Relationship Scores")
                st.caption("Simple scores indicating potential relationship strength (requires preprocessing). Higher is generally better.")

                try:
                    with st.spinner("Calculating feature scores..."):
                        # Prepare data for scoring (impute, encode) - Use simpler encoding for scores
                        df_scored = df_xai[features_xai + [target_xai]].copy()
                        df_scored.dropna(subset=[target_xai], inplace=True) # Drop rows where target is NA

                        # Impute features
                        num_imputer_xai = SimpleImputer(strategy='median')
                        cat_imputer_xai = SimpleImputer(strategy='most_frequent')

                        if numeric_features_xai:
                            df_scored[numeric_features_xai] = num_imputer_xai.fit_transform(df_scored[numeric_features_xai])
                        if categorical_features_xai:
                             df_scored[categorical_features_xai] = cat_imputer_xai.fit_transform(df_scored[categorical_features_xai])
                             # Simple ordinal encoding for categorical features for scoring
                             encoder_ord = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                             df_scored[categorical_features_xai] = encoder_ord.fit_transform(df_scored[categorical_features_xai])
                        if bool_features_xai: # Convert bool to int (0/1)
                             for bf in bool_features_xai: df_scored[bf] = df_scored[bf].astype(int)

                        # Target encoding/preparation
                        if is_target_classification:
                            le = LabelEncoder()
                            y_scored = le.fit_transform(df_scored[target_xai])
                            target_classes = le.classes_ # Store classes for later use
                        else: # Regression
                            y_scored = pd.to_numeric(df_scored[target_xai], errors='coerce')
                            # Align X and y after potential coercion
                            valid_target_idx = y_scored.notna()
                            df_scored = df_scored[valid_target_idx]
                            y_scored = y_scored[valid_target_idx]


                        X_scored = df_scored[features_xai]
                        scores_data = []

                        if not X_scored.empty and not y_scored.empty:
                            # Select appropriate scoring function
                            if is_target_classification:
                                # Use f_classif (ANOVA F-value) for numeric features
                                if numeric_features_xai:
                                    f_scores, _ = f_classif(X_scored[numeric_features_xai], y_scored)
                                    scores_data.extend([{"Feature": f, "Score (F-Classif)": s, "Type": "Numeric"} for f, s in zip(numeric_features_xai, f_scores)])
                                # Use mutual_info_classif for categorical/ordinal/bool features
                                mi_features = categorical_features_xai + bool_features_xai
                                if mi_features:
                                    mi_scores = mutual_info_classif(X_scored[mi_features], y_scored, discrete_features=True, random_state=42)
                                    scores_data.extend([{"Feature": f, "Score (Mutual Info)": s, "Type": "Categorical/Bool"} for f, s in zip(mi_features, mi_scores)])

                            else: # Regression
                                # Use f_regression for numeric features
                                if numeric_features_xai:
                                    f_scores_reg, _ = f_regression(X_scored[numeric_features_xai], y_scored)
                                    scores_data.extend([{"Feature": f, "Score (F-Regr)": s, "Type": "Numeric"} for f, s in zip(numeric_features_xai, f_scores_reg)])
                                # Use mutual_info_regression for categorical/ordinal/bool features
                                mi_features = categorical_features_xai + bool_features_xai
                                if mi_features:
                                     mi_scores_reg = mutual_info_regression(X_scored[mi_features], y_scored, discrete_features=True, random_state=42)
                                     scores_data.extend([{"Feature": f, "Score (Mutual Info)": s, "Type": "Categorical/Bool"} for f, s in zip(mi_features, mi_scores_reg)])

                            if scores_data:
                                scores_df = pd.DataFrame(scores_data)
                                scores_df = scores_df.sort_values(by=scores_df.columns[1], ascending=False).reset_index(drop=True)
                                st.dataframe(scores_df, use_container_width=True)
                            else: st.info("Could not calculate scores for any features.")
                        else: st.warning("Not enough valid data to calculate feature scores after preprocessing.")

                except Exception as score_err:
                    st.error(f"Error calculating feature scores: {score_err}")
                    st.error(traceback.format_exc())


                st.subheader("Visual Insights")

                # Limit number of plots for performance
                max_plots = 15
                num_plots_so_far = 0

                if is_target_classification:
                     st.markdown("**Numeric Features vs. Target Classes**")
                     for i, col in enumerate(numeric_features_xai):
                         if num_plots_so_far >= max_plots: break
                         try:
                              fig, ax = plt.subplots()
                              sns.histplot(data=df_xai, x=col, hue=target_xai, kde=True, ax=ax, element="step")
                              ax.set_title(f"Distribution of '{col}' by '{target_xai}'")
                              plt.tight_layout(); st.pyplot(fig)
                              num_plots_so_far += 1
                         except Exception as e: st.warning(f"Plot failed for {col}: {e}")
                         finally: plt.close('all') # Close figures

                     st.markdown("**Categorical/Boolean Features vs. Target Classes**")
                     for i, col in enumerate(categorical_features_xai + bool_features_xai):
                          if num_plots_so_far >= max_plots: break
                          if df_xai[col].nunique() < 30: # Only plot categoricals with reasonable cardinality
                              try:
                                   fig, ax = plt.subplots()
                                   # Create counts and normalize within each category for comparison
                                   plot_df = df_xai.groupby(col)[target_xai].value_counts(normalize=True).mul(100).rename('percent').reset_index()
                                   # Sort bars for better readability maybe?
                                   sns.barplot(data=plot_df, y=col, x='percent', hue=target_xai, ax=ax, orient='h')
                                   ax.set_title(f"Proportion of '{target_xai}' within '{col}'")
                                   ax.legend(title=target_xai, bbox_to_anchor=(1.05, 1), loc='upper left')
                                   plt.tight_layout(); st.pyplot(fig)
                                   num_plots_so_far += 1
                              except Exception as e: st.warning(f"Plot failed for {col}: {e}")
                              finally: plt.close('all')


                else: # Regression Target
                     st.markdown("**Numeric Features vs. Target**")
                     for i, col in enumerate(numeric_features_xai):
                          if num_plots_so_far >= max_plots: break
                          try:
                               fig, ax = plt.subplots()
                               sns.scatterplot(data=df_xai, x=col, y=target_xai, ax=ax, alpha=0.5)
                               # Add a regression line? Maybe too much computation
                               # sns.regplot(data=df_xai, x=col, y=target_xai, ax=ax, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
                               ax.set_title(f"'{target_xai}' vs '{col}'")
                               plt.tight_layout(); st.pyplot(fig)
                               num_plots_so_far += 1
                          except Exception as e: st.warning(f"Plot failed for {col}: {e}")
                          finally: plt.close('all')


                     st.markdown("**Categorical/Boolean Features vs. Target**")
                     for i, col in enumerate(categorical_features_xai + bool_features_xai):
                          if num_plots_so_far >= max_plots: break
                          if df_xai[col].nunique() < 30:
                              try:
                                   fig, ax = plt.subplots()
                                   # Box plot or Violin plot
                                   sns.boxplot(data=df_xai, x=col, y=target_xai, ax=ax, showfliers=False, order=df_xai[col].value_counts().index[:30]) # Order by count, limit cats
                                   ax.set_title(f"Distribution of '{target_xai}' by '{col}'")
                                   plt.xticks(rotation=45, ha='right'); plt.tight_layout(); st.pyplot(fig)
                                   num_plots_so_far += 1
                              except Exception as e: st.warning(f"Plot failed for {col}: {e}")
                              finally: plt.close('all')

                if num_plots_so_far == 0:
                     st.info("No suitable plots generated based on selected target and available features.")

            else:
                 st.info("Select a target variable above to see insights.")

    # --- Tab: Model Advisor ---
    with tab_advisor:
        st.header("🧠 ML Model Selection Advisor")
        st.markdown("Get simple model suggestions based on your data characteristics.")

        df_adv = st.session_state.df
        if df_adv is None or df_adv.empty: st.warning("Load data first.")
        else:
            cols_adv = df_adv.columns.tolist()
            num_cols_a, cat_cols_a, dt_cols_a, bool_cols_a = get_column_types(df_adv)

            target_options_adv = [None] + num_cols_a + cat_cols_a + bool_cols_a
            target_adv = st.selectbox("Select Target Variable for Advice:", target_options_adv,
                                       index=target_options_adv.index(st.session_state.ml_lite_target) if st.session_state.ml_lite_target in target_options_adv else 0,
                                       key="adv_target")
            if target_adv != st.session_state.ml_lite_target:
                st.session_state.ml_lite_target = target_adv
                st.rerun()

            if target_adv:
                target_series_adv = df_adv[target_adv]
                is_target_numeric_adv = pd.api.types.is_numeric_dtype(target_series_adv.dtype) and target_series_adv.nunique() > 20 # Heuristic
                is_target_classification_adv = not is_target_numeric_adv

                n_rows = len(df_adv)
                n_features = len(cols_adv) - 1

                st.subheader("Recommendations")

                if is_target_classification_adv:
                    st.markdown("**Problem Type:** Classification")
                    st.markdown("**Suggested Models:**")
                    st.markdown("- **Logistic Regression:** Good linear baseline, interpretable.")
                    st.markdown("- **Random Forest Classifier:** Often high performance, handles non-linearities, less prone to overfitting than single decision trees.")
                    st.markdown("- **K-Nearest Neighbors (KNN):** Simple instance-based learner, can capture complex boundaries.")
                    if n_rows < 50000: # SVC can be slow on large datasets
                        st.markdown("- **Support Vector Classifier (SVC):** Effective in high dimensions, but can be computationally intensive.")
                    if any(df_adv[f].dtype == 'object' for f in cat_cols_a): # Suggest Naive Bayes if text-like object columns exist
                         st.markdown("- **Naive Bayes (e.g., MultinomialNB):** Often good for text classification if features are processed appropriately (e.g., TF-IDF).")
                    st.markdown("**Considerations:** *Preprocessing (scaling for KNN/SVC, encoding categories) is crucial. Try simpler models first.*")

                elif is_target_numeric_adv:
                    st.markdown("**Problem Type:** Regression")
                    st.markdown("**Suggested Models:**")
                    st.markdown("- **Linear Regression:** Simple baseline, interpretable coefficients.")
                    st.markdown("- **Ridge / Lasso Regression:** Linear models with regularization, good if features might be correlated.")
                    st.markdown("- **Random Forest Regressor:** Powerful non-linear model, often robust.")
                    if n_rows < 50000:
                         st.markdown("- **Support Vector Regressor (SVR):** Non-linear regression, can be effective but potentially slow.")
                    st.markdown("- **Gradient Boosting Regressor (e.g., XGBoost, LightGBM):** (Requires separate install) Often state-of-the-art performance, but more complex tuning.")
                    st.markdown("**Considerations:** *Feature scaling is important for many regression models. Check for outliers and linearity assumptions.*")
                else:
                    st.warning("Could not determine clear problem type for target.")

                st.caption(f"Advice based on target type and basic data shape ({n_rows} rows, {n_features} features). Does not account for specific data distributions or complex feature interactions.")

            else:
                st.info("Select a target variable to get model advice.")

# --- Handling Initial State (No DataFrame Loaded) ---
elif st.session_state.data_source is None:
    st.info("👈 Choose data source from sidebar.")
else: # Source selected but df is None
    if st.session_state.data_source == 'Upload File (CSV/Excel)': st.info("👈 Upload file.")
    elif st.session_state.data_source == 'Connect to Database': st.info("👈 Complete DB connection & fetch.")
