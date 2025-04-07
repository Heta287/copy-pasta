# Final Complete app.py (with DB Config File, ML Tab, Boolean Display Fix)

import streamlit as st
import pandas as pd
import pyodbc
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ydata_profiling import ProfileReport # Updated name for pandas-profiling
import sweetviz as sv
import streamlit.components.v1 as components # For embedding HTML reports
import os
import json # For reading DB config
import traceback

# --- ML Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer

# --- Constants ---
DB_CONFIG_FILE = "db_config.json"
DEFAULT_ODBC_DRIVER = "{ODBC Driver 17 for SQL Server}" # Change if needed
DEFAULT_TRUSTED_CONNECTION = "yes" # Change to "no" if using UID/PWD by default
# DEFAULT_UID = "YOUR_DEFAULT_USERNAME" # Optional: Define if not using trusted connection
# DEFAULT_PWD = "YOUR_DEFAULT_PASSWORD" # Optional: Define if not using trusted connection


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

# --- Action Definitions (for cleaner UI selection) ---
ACTION_CATEGORIES = {
    "Basic Info": ["Show Shape", "Show Columns & Types", "Show Basic Statistics (Describe)", "Show Missing Values", "Show Unique Values (Categorical)", "Count Values (Categorical)"],
    "Data Cleaning": ["Drop Columns", "Rename Columns", "Handle Missing Data", "Drop Duplicate Rows", "Change Data Type", "String Manipulation (Trim, Case, Replace)", "Extract from Text (Regex)", "Date Component Extraction (Year, Month...)"],
    "Data Transformation": ["Filter Data", "Sort Data", "Select Columns", "Create Calculated Column (Basic Arithmetic)", "Bin Numeric Data (Cut)", "One-Hot Encode Categorical Column", "Pivot (Simple)" , "Melt (Unpivot)"],
    "Aggregation & Analysis": ["Calculate Single Aggregation (Sum, Mean...)", "Group By & Aggregate", "Calculate Rolling Window Statistics", "Calculate Cumulative Statistics", "Rank Data"],
    "Visualization": ["Plot Histogram (Numeric)", "Plot Density Plot (Numeric)", "Plot Count Plot (Categorical)", "Plot Bar Chart (Aggregated)", "Plot Line Chart", "Plot Scatter Plot", "Plot Box Plot (Numeric vs Cat)", "Plot Correlation Heatmap (Numeric)"]
}

# --- Helper Functions (Keep existing helpers: get_databases, get_tables, fetch_data_from_db, load_file_data, get_column_types, convert_df_to_csv, convert_df_to_excel, save_plot_to_bytes, generate_profile_report, generate_sweetviz_report) ---

# Modify get_databases and get_tables to use the loaded DB_SERVERS
@st.cache_data(show_spinner="Connecting to database to fetch names...")
def get_databases(server_name):
    """Fetches list of databases from a given server using config."""
    if not server_name or server_name not in DB_SERVERS:
        return []
    server_address = DB_SERVERS[server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # Add UID/PWD if not using trusted connection (optional based on defaults)
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no':
    #      conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    databases = []
    try:
        # Use connection attributes for timeout which is more reliable across drivers
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}
        with pyodbc.connect(conn_str, timeout=10, attrs_before=conn_attrs) as conn:
            cursor = conn.cursor()
            # This query might need adjustment based on DB permissions/version
            cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb');")
            databases = [row.name for row in cursor.fetchall()]
    except pyodbc.Error as ex:
        st.error(f"Error connecting to {server_name} ({server_address}) to list databases: {ex}. Check config/network/permissions.")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching databases: {e}")
    return databases

@st.cache_data(show_spinner="Fetching table names...")
def get_tables(server_name, db_name):
    """Fetches list of tables from a given database using config."""
    if not server_name or not db_name or server_name not in DB_SERVERS:
        return []
    server_address = DB_SERVERS[server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};DATABASE={db_name};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no':
    #      conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    tables = []
    try:
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 10}
        with pyodbc.connect(conn_str, timeout=10, attrs_before=conn_attrs) as conn:
            cursor = conn.cursor()
            # Query for user tables
            cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME")
            tables = [f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}" for row in cursor.fetchall()]
    except pyodbc.Error as ex:
        st.error(f"Error connecting to {db_name} on {server_name} ({server_address}) to list tables: {ex}. Check config/permissions.")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching tables: {e}")
    return tables

@st.cache_data(show_spinner="Fetching data from database...")
def fetch_data_from_db(_server_name, _db_name, _sql_query):
    """Connects to DB and fetches data based on SQL query using config."""
    df = pd.DataFrame()
    if not _server_name or not _db_name or _server_name not in DB_SERVERS:
        st.error("Invalid server selection for fetching data.")
        return df
    server_address = DB_SERVERS[_server_name]
    conn_str = (f"DRIVER={{{DEFAULT_ODBC_DRIVER}}};SERVER={server_address};DATABASE={_db_name};"
                f"Trusted_Connection={DEFAULT_TRUSTED_CONNECTION};")
    # if DEFAULT_TRUSTED_CONNECTION.lower() == 'no':
    #      conn_str += f"UID={DEFAULT_UID};PWD={DEFAULT_PWD};"
    try:
        # Set longer timeout for query execution itself
        conn_attrs = {pyodbc.SQL_ATTR_LOGIN_TIMEOUT: 15, pyodbc.SQL_ATTR_CONNECTION_TIMEOUT: 60} # Login timeout + query timeout
        with pyodbc.connect(conn_str, timeout=60, attrs_before=conn_attrs) as conn:
            df = pd.read_sql(_sql_query, conn)
            # Attempt basic type conversion (e.g., infer objects) after fetch
            if not df.empty:
                df = df.convert_dtypes() # More modern way than infer_objects
    except pyodbc.Error as ex:
        st.error(f"Database Error: {ex}. Failed to execute query. Check syntax and permissions.")
        st.code(_sql_query, language='sql') # Show the failed query
    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching: {e}")
        st.error(traceback.format_exc())
    return df

@st.cache_data
def load_file_data(uploaded_file):
    """Loads data from uploaded CSV or Excel file."""
    df = None
    try:
        file_name = uploaded_file.name
        st.info(f"Loading file: '{file_name}'...")
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xls', '.xlsx')):
            excel_file = pd.ExcelFile(uploaded_file)
            if len(excel_file.sheet_names) > 1:
                selected_sheet = st.selectbox(f"Select sheet to load from '{file_name}':", excel_file.sheet_names, key=f"sheet_{file_name}")
                if selected_sheet:
                    df = excel_file.parse(selected_sheet)
            elif len(excel_file.sheet_names) == 1:
                df = excel_file.parse(excel_file.sheet_names[0])
            else:
                st.error(f"Excel file '{file_name}' contains no sheets.")
        else:
            st.error("Unsupported file format. Please upload CSV or Excel (.xls, .xlsx).")

        # Apply convert_dtypes after loading
        if df is not None and not df.empty:
             df = df.convert_dtypes()

    except Exception as e:
        st.error(f"Error processing uploaded file '{uploaded_file.name}': {e}")
        st.error(traceback.format_exc()) # More detailed error for debugging
        df = None # Ensure df is None on error
    return df

def get_column_types(df):
    """Helper to categorize columns using pandas modern dtypes."""
    if df is None:
        return [], [], []
    try:
        numeric_cols = df.select_dtypes(include=['number', 'boolean']).columns.tolist() # Include pandas boolean
        # Exclude boolean from categorical explicitly if already in numeric
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Boolean might be inferred as object sometimes, check explicitly
        bool_like_object_cols = [col for col in categorical_cols if df[col].dropna().apply(lambda x: str(x).lower() in ['true', 'false', '1', '0', 'yes', 'no']).all()]
        categorical_cols = [col for col in categorical_cols if col not in bool_like_object_cols]
        numeric_cols.extend(bool_like_object_cols) # Treat bool-like objects as numeric/bool for simplicity here

        datetime_cols = df.select_dtypes(include=['datetime', 'timedelta']).columns.tolist()

        # Refine categorical: exclude columns with very high cardinality (likely IDs or free text)
        potential_cat_cols = categorical_cols.copy()
        # Increase threshold slightly, maybe based on number of rows too?
        max_unique_ratio = 0.6 # If > 60% unique values
        min_rows_for_ratio_check = 50 # Only apply ratio check if enough rows
        rows = len(df)

        if rows > min_rows_for_ratio_check:
            for col in potential_cat_cols:
                 try:
                    # Use dropna=False for ratio calculation? Debatable. Let's keep dropna=True for categorical meaning.
                    unique_count = df[col].nunique() # Defaults to dropna=True
                    unique_ratio = unique_count / rows
                    # Also check absolute unique count - maybe if > 100 unique cats, it's not useful for plots
                    if unique_ratio > max_unique_ratio or unique_count > 100:
                         if col in categorical_cols:
                             categorical_cols.remove(col)
                 except TypeError: # Handle non-hashable types if any
                      if col in categorical_cols: categorical_cols.remove(col)
                 except Exception: # Catch other potential errors
                      if col in categorical_cols: categorical_cols.remove(col)

        # Ensure no overlap after refinement
        numeric_cols = list(set(numeric_cols))
        categorical_cols = list(set(categorical_cols) - set(numeric_cols) - set(datetime_cols))
        datetime_cols = list(set(datetime_cols) - set(numeric_cols))


        return numeric_cols, categorical_cols, datetime_cols
    except Exception as e:
        st.warning(f"Could not accurately determine column types: {e}")
        # Fallback: return basic types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        other_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        return numeric_cols, other_cols, []

def convert_df_to_csv(df):
   """Converts DataFrame to CSV bytes for download."""
   return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Converts DataFrame to Excel bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sanitize sheet name if needed (Excel has restrictions)
        safe_sheet_name = "Sheet1" # Use a fixed safe name
        df.to_excel(writer, index=False, sheet_name=safe_sheet_name)
    return output.getvalue()

def save_plot_to_bytes(fig):
    """Saves a matplotlib figure to bytes for download and closes the figure."""
    img_bytes = io.BytesIO()
    try:
        if fig is not None and isinstance(fig, plt.Figure):
             # Use bbox_inches='tight' to prevent labels cutoff
             fig.savefig(img_bytes, format='png', bbox_inches='tight', dpi=150) # Increase DPI for better quality
             plt.close(fig) # IMPORTANT: Close the figure to free memory
             img_bytes.seek(0)
             return img_bytes.getvalue()
        else:
            st.warning("Invalid figure object passed for saving.")
            return None
    except Exception as e:
         st.error(f"Error saving plot: {e}")
         if fig is not None and isinstance(fig, plt.Figure):
              plt.close(fig) # Ensure figure is closed even on error
         return None

# --- Caching functions for EDA Reports ---
@st.cache_data(show_spinner="Generating YData Profile Report (this can take a while)...")
def generate_profile_report(_df, _title="Data Profile Report"):
    """Generates YData Profiling report HTML."""
    if _df is None or _df.empty:
        st.warning("Cannot generate profile report on empty data.")
        return None
    try:
        # Further reduce complexity for speed
        profile = ProfileReport(_df, title=_title,
                                minimal=True, # Use minimal mode
                                # explorative=True, # Minimal overrides explorative
                                # Disable more components if still slow
                                # samples={"head": 0, "tail": 0},
                                # correlations=None,
                                # missing_diagrams=None,
                                # duplicates=None,
                                )
        return profile.to_html()
    except Exception as e:
        st.error(f"Error generating YData profile report: {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_data(show_spinner="Generating Sweetviz Report (this can take a while)...")
def generate_sweetviz_report(_df):
     """Generates Sweetviz report HTML."""
     if _df is None or _df.empty:
        st.warning("Cannot generate Sweetviz report on empty data.")
        return None
     try:
         report = sv.analyze(_df)
         # Sweetviz show_html might try file saving; capture HTML string instead
         # Use internal method if available or save/read temp file as workaround
         # Assuming show_html can return the string directly based on parameters
         html_report = report.show_html(filepath=None, open_browser=False, layout='vertical', scale=None)
         return html_report

     except NotImplementedError as nie:
         st.error(f"Sweetviz feature not implemented or failed: {nie}. This can sometimes happen with specific data types.")
         return None
     except Exception as e:
         st.error(f"Error generating Sweetviz report: {e}")
         st.error(traceback.format_exc())
         return None

# --- ML Helper Functions ---
def get_ml_pipeline(numeric_features, categorical_features, numeric_imputer_strategy, cat_imputer_strategy, cat_fill_value, scaler_option, encoder_option):
    """Creates a ColumnTransformer pipeline for preprocessing."""
    numeric_steps = [('imputer', SimpleImputer(strategy=numeric_imputer_strategy))]
    if scaler_option == "StandardScaler": numeric_steps.append(('scaler', StandardScaler()))
    elif scaler_option == "MinMaxScaler": numeric_steps.append(('scaler', MinMaxScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_steps = [('imputer', SimpleImputer(strategy=cat_imputer_strategy, fill_value=cat_fill_value if cat_fill_value else 'missing'))]
    # Add encoder based on option (currently only OneHotEncoder)
    if encoder_option == "OneHotEncoder":
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))) # sparse=False often easier downstream

    categorical_pipeline = Pipeline(steps=categorical_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    return preprocessor

def get_model(model_name):
    """Returns an untrained scikit-learn model instance based on name."""
    # Regression Models
    if model_name == "Linear Regression": return LinearRegression()
    if model_name == "Ridge Regression": return Ridge(random_state=42)
    if model_name == "Lasso Regression": return Lasso(random_state=42)
    if model_name == "Decision Tree Regressor": return DecisionTreeRegressor(random_state=42)
    if model_name == "Random Forest Regressor": return RandomForestRegressor(random_state=42, n_estimators=50, n_jobs=-1) # Use more cores

    # Classification Models
    if model_name == "Logistic Regression": return LogisticRegression(random_state=42, max_iter=200, n_jobs=-1)
    if model_name == "Decision Tree Classifier": return DecisionTreeClassifier(random_state=42)
    if model_name == "Random Forest Classifier": return RandomForestClassifier(random_state=42, n_estimators=50, n_jobs=-1)
    if model_name == "Support Vector Classifier (Linear)": return SVC(kernel='linear', probability=True, random_state=42)
    if model_name == "K-Nearest Neighbors": return KNeighborsClassifier(n_jobs=-1)

    st.error(f"Unknown model name selected: {model_name}")
    return None


# --- Initialize Session State ---
st.session_state.setdefault('data_source', None)
st.session_state.setdefault('df', None)
st.session_state.setdefault('db_params', {'server': None, 'database': None, 'table': None})
st.session_state.setdefault('generated_code', "") # For Wizard
st.session_state.setdefault('result_type', None) # For Wizard
st.session_state.setdefault('uploaded_file_state', None)
st.session_state.setdefault('selected_category', list(ACTION_CATEGORIES.keys())[0])
st.session_state.setdefault('show_profile_report', False)
st.session_state.setdefault('show_sweetviz_report', False)
st.session_state.setdefault('current_action_result_display', None)
# ML State Variables
st.session_state.setdefault('ml_problem_type', None)
st.session_state.setdefault('ml_features', [])
st.session_state.setdefault('ml_target', None)
st.session_state.setdefault('ml_pipeline', None) # Stores the fitted sklearn pipeline
st.session_state.setdefault('ml_evaluation', {}) # Stores metrics, plots
st.session_state.setdefault('ml_code', "") # Code for ML process
st.session_state.setdefault('ml_model_name', None)


# --- App Layout ---
st.set_page_config(layout="wide", page_title="Data Analysis & ML Wizard")
st.title("📊 Data Analysis & ML Wizard 🧙‍♂️")
st.markdown("Connect to data, explore with the Wizard, or build basic ML models.")

# --- Sidebar for Data Connection & Session Control ---
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

    current_data_source_index = None
    if st.session_state.data_source == 'Upload File (CSV/Excel)': current_data_source_index = 0
    elif st.session_state.data_source == 'Connect to Database' and db_connection_available: current_data_source_index = 1

    data_source_option = st.radio(
        "Choose your data source:", connection_options,
        key='data_source_radio', index=current_data_source_index,
        on_change=lambda: st.session_state.update(df=None, generated_code="", result_type=None, current_action_result_display=None, show_profile_report=False, show_sweetviz_report=False, ml_problem_type=None, ml_features=[], ml_target=None, ml_pipeline=None, ml_evaluation={}, ml_code="", ml_model_name=None)
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
                st.session_state.df = None # Clear before loading
                # Reset all results/ML state
                st.session_state.generated_code = "" ; st.session_state.result_type = None ; st.session_state.current_action_result_display = None
                st.session_state.show_profile_report = False ; st.session_state.show_sweetviz_report = False
                st.session_state.ml_problem_type = None ; st.session_state.ml_features = [] ; st.session_state.ml_target = None
                st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {} ; st.session_state.ml_code = "" ; st.session_state.ml_model_name=None
                # Load data
                st.session_state.df = load_file_data(uploaded_file) # Function now applies convert_dtypes

                if st.session_state.df is not None:
                    st.success(f"Successfully loaded `{uploaded_file.name}`")
                    st.rerun()
                else:
                    st.session_state.uploaded_file_state = None

    # --- Database Connection Section ---
    elif st.session_state.data_source == 'Connect to Database':
        st.subheader("🗄️ Connect to Database")
        if not db_connection_available:
            st.error(f"Database connection disabled. Ensure '{DB_CONFIG_FILE}' exists and is configured correctly.")
        else:
            server_names = list(DB_SERVERS.keys())
            current_server = st.session_state.db_params.get('server')
            server_index = server_names.index(current_server) if current_server in server_names else None
            selected_server = st.selectbox(
                "1. Select Server", options=server_names, index=server_index, placeholder="Choose a server...", key="db_server_select"
            )
            if selected_server != current_server:
                 st.session_state.db_params['server'] = selected_server
                 st.session_state.db_params['database'] = None ; st.session_state.db_params['table'] = None ; st.session_state.df = None
                 st.rerun()

            selected_db = None
            current_server = st.session_state.db_params.get('server') # Get updated server
            if current_server:
                available_dbs = get_databases(current_server)
                current_db = st.session_state.db_params.get('database')
                db_index = available_dbs.index(current_db) if current_db in available_dbs else None
                if available_dbs:
                    selected_db = st.selectbox(
                        "2. Select Database", options=available_dbs, index=db_index, placeholder="Choose a database...", key="db_select"
                    )
                    if selected_db != current_db:
                        st.session_state.db_params['database'] = selected_db
                        st.session_state.db_params['table'] = None ; st.session_state.df = None
                        st.rerun()
                else: st.session_state.db_params['database'] = None

            selected_table = None
            current_server = st.session_state.db_params.get('server')
            current_db = st.session_state.db_params.get('database')
            if current_server and current_db:
                available_tables = get_tables(current_server, current_db)
                current_table = st.session_state.db_params.get('table')
                table_index = available_tables.index(current_table) if current_table in available_tables else None
                if available_tables:
                    selected_table = st.selectbox(
                        "3. Select Table", options=available_tables, index=table_index, placeholder="Choose a table...", key="db_table_select"
                    )
                    if selected_table != current_table:
                         st.session_state.db_params['table'] = selected_table
                         st.session_state.df = None
                         st.rerun()
                else: st.session_state.db_params['table'] = None

            current_table = st.session_state.db_params.get('table')
            if current_table:
                query_method = st.radio("4. Fetch Method", ("Select TOP 1000 Rows", "Custom SQL Query"), key="db_query_method")
                sql_query = ""
                if query_method == "Select TOP 1000 Rows":
                     table_parts = current_table.split('.')
                     quoted_table = f"[{table_parts[0]}].[{table_parts[1]}]" if len(table_parts) == 2 else f"[{current_table}]"
                     sql_query = f"SELECT TOP 1000 * FROM {quoted_table};"
                     st.text_area("Generated SQL:", value=sql_query, height=100, disabled=True, key="db_sql_display")
                else:
                    default_custom_sql = f"SELECT * FROM {current_table};" if current_table else "SELECT * FROM your_table;"
                    st.session_state.setdefault('custom_sql_input', default_custom_sql)
                    sql_query = st.text_area("Enter your SQL Query:", value=st.session_state.custom_sql_input, height=150, key="db_sql_custom", on_change=lambda: st.session_state.update(custom_sql_input=st.session_state.db_sql_custom))

                if st.button("Fetch Data from Database", key="db_fetch_button"):
                    current_sql_to_run = sql_query
                    current_server_fetch = st.session_state.db_params.get('server')
                    current_database_fetch = st.session_state.db_params.get('database')
                    if current_sql_to_run and current_server_fetch and current_database_fetch:
                        # Reset state before fetching
                        st.session_state.df = None ; st.session_state.generated_code = "" ; st.session_state.result_type = None
                        st.session_state.show_profile_report = False ; st.session_state.show_sweetviz_report = False
                        st.session_state.current_action_result_display = None ; st.session_state.uploaded_file_state = None
                        st.session_state.ml_problem_type = None ; st.session_state.ml_features = [] ; st.session_state.ml_target = None
                        st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {} ; st.session_state.ml_code = "" ; st.session_state.ml_model_name=None

                        fetched_df = fetch_data_from_db(current_server_fetch, current_database_fetch, current_sql_to_run)
                        st.session_state.df = fetched_df # Assign fetched data (already dtype converted)

                        if fetched_df is not None and not fetched_df.empty:
                            st.success(f"Successfully fetched data.") ; st.rerun()
                        elif fetched_df is not None and fetched_df.empty:
                            st.warning("Query executed successfully but returned no data.") ; st.rerun()
                    else:
                        st.warning("Ensure Server, Database, and Table/Query are selected/entered.")


# --- Main Panel: Display Data, EDA, Wizard, ML ---
if st.session_state.df is not None:
    st.header("Preview of Loaded Data")
    # FIX 3: Convert boolean columns to string *only for display* in st.dataframe to avoid checkboxes
    df_display = st.session_state.df.copy()
    try:
        # Use modern pandas boolean type if present
        bool_cols = df_display.select_dtypes(include='boolean').columns # Pandas nullable boolean
        for col in bool_cols:
            df_display[col] = df_display[col].astype(str) # Convert to string for display
        # Also check older bool type if necessary
        old_bool_cols = df_display.select_dtypes(include=bool).columns
        for col in old_bool_cols:
             if col not in bool_cols: # Avoid double conversion
                 df_display[col] = df_display[col].astype(str)
    except Exception as display_conv_err:
        st.warning(f"Could not convert boolean columns for display: {display_conv_err}")

    st.dataframe(df_display, height=300, use_container_width=True)

    # --- Tabs for different functionalities ---
    tab_titles = ["Automated EDA", "Data Analysis Wizard", "Machine Learning"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    # --- Tab 1: Automated EDA Reports ---
    with tab1:
        st.subheader("📊 Automated EDA Reports")
        st.markdown("Generate comprehensive reports for quick data understanding.")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
             if st.button("Generate YData Profile Report", key="profile_report_btn", use_container_width=True):
                 st.session_state.show_profile_report = True
                 st.session_state.show_sweetviz_report = False
                 st.session_state.current_action_result_display = None
                 st.rerun()
        with report_col2:
            if st.button("Generate Sweetviz Report", key="sweetviz_report_btn", use_container_width=True):
                st.session_state.show_sweetviz_report = True
                st.session_state.show_profile_report = False
                st.session_state.current_action_result_display = None
                st.rerun()

        report_placeholder = st.container()
        with report_placeholder:
            if st.session_state.get('show_profile_report', False):
                report_title = f"Profile Report for {st.session_state.uploaded_file_state[0]}" if st.session_state.uploaded_file_state else "Profile Report"
                report_html = generate_profile_report(st.session_state.df, report_title)
                if report_html:
                    with st.expander("YData Profile Report", expanded=True): components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate YData Profile Report.")

            if st.session_state.get('show_sweetviz_report', False):
                report_html = generate_sweetviz_report(st.session_state.df)
                if report_html:
                     with st.expander("Sweetviz Report", expanded=True): components.html(report_html, height=600, scrolling=True)
                else: st.warning("Could not generate Sweetviz Report.")


    # --- Tab 2: Data Analysis Wizard ---
    with tab2:
        st.header("🧙‍♂️ Data Analysis Wizard")
        st.markdown("Perform common data tasks without coding. See results and generated Python code below.")

        df_wizard = st.session_state.df # Use the main df for the wizard actions
        if df_wizard is None or df_wizard.empty:
             st.warning("No data loaded for the wizard.")
        else:
            numeric_cols_wiz, categorical_cols_wiz, datetime_cols_wiz = get_column_types(df_wizard)
            all_cols_wiz = df_wizard.columns.tolist()

            st.subheader("1. Select Action Category")
            selected_category_index = list(ACTION_CATEGORIES.keys()).index(st.session_state.selected_category)
            selected_category = st.radio("Category:", list(ACTION_CATEGORIES.keys()), index=selected_category_index, horizontal=True, key="category_radio_wizard")
            if selected_category != st.session_state.selected_category:
                st.session_state.selected_category = selected_category
                st.rerun()

            st.subheader("2. Choose Specific Action & Configure")
            action_options = ACTION_CATEGORIES[selected_category]
            st.session_state.setdefault(f'selected_action_{selected_category}', action_options[0])
            selected_action_key = f'selected_action_{selected_category}'
            current_action_in_state = st.session_state.get(selected_action_key, action_options[0])
            selected_action_index = action_options.index(current_action_in_state) if current_action_in_state in action_options else 0
            action = st.selectbox(f"Select Action in '{selected_category}':", action_options, index=selected_action_index, key=f"action_select_{selected_category}")
            st.session_state[selected_action_key] = action

            # --- Wizard Action Implementation Block ---
            code_string = "# Select an action and configure options"
            action_configured = False # Flag to check if options are set

            try:
                # === Basic Info ===
                if action == "Show Shape":
                    code_string = "result_data = df.shape"
                    action_configured = True
                elif action == "Show Columns & Types":
                    # Use .info() in a buffer to get a nice string representation
                    code_string = """
import io
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
result_data = info_str
"""
                    action_configured = True
                elif action == "Show Basic Statistics (Describe)":
                    # Use updated column types
                    sel_cols_describe = st.multiselect("Select Columns (Optional, default=all numeric/datetime/bool):", all_cols_wiz, default=numeric_cols_wiz + datetime_cols_wiz, key=f"desc_{action}")
                    if sel_cols_describe:
                        code_string = f"# Describe specific columns\nresult_data = df[{sel_cols_describe}].describe(include='all')" # include='all' for mixed types
                    else: # If none selected, describe all suitable types
                        code_string = f"# Describe all suitable columns\nresult_data = df.describe(include='all')"
                    action_configured = True # Always possible
                elif action == "Show Missing Values":
                    code_string = "missing_counts = df.isnull().sum()\nresult_data = missing_counts[missing_counts > 0].reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'}).sort_values('Missing Count', ascending=False)"
                    action_configured = True
                elif action == "Show Unique Values (Categorical)":
                    if not categorical_cols_wiz: st.warning("No categorical columns identified for this action.")
                    cat_col_unique = st.selectbox("Select Categorical Column:", categorical_cols_wiz, key=f"unique_{action}")
                    if cat_col_unique:
                        max_uniques_show = 100 # Limit display
                        code_string = f"unique_vals = df['{cat_col_unique}'].unique()\nnum_uniques = len(unique_vals)\nif num_uniques > {max_uniques_show}: result_data = f'{{num_uniques}} unique values (showing first {max_uniques_show}):\\n' + '\\n'.join(map(str, unique_vals[:{max_uniques_show}]))\nelse: result_data = pd.DataFrame(unique_vals, columns=['Unique Values']).dropna()"
                        action_configured = True
                elif action == "Count Values (Categorical)":
                    if not categorical_cols_wiz: st.warning("No categorical columns identified for this action.")
                    cat_col_count = st.selectbox("Select Categorical Column:", categorical_cols_wiz, key=f"count_{action}")
                    if cat_col_count:
                        normalize = st.checkbox("Show as Percentage (%)", key=f"count_norm_{action}")
                        # Include NA in counts by default maybe? dropna=False
                        code_string = f"counts = df['{cat_col_count}'].value_counts(normalize={normalize}, dropna=False).reset_index()\ncounts.columns = ['{cat_col_count}', 'Percentage' if {normalize} else 'Count']\nresult_data = counts"
                        action_configured = True

                # === Data Cleaning ===
                elif action == "Drop Columns":
                     drop_cols = st.multiselect("Select Columns to Drop:", all_cols_wiz, key=f"drop_{action}")
                     if drop_cols:
                         code_string = f"result_data = df.drop(columns={drop_cols})"
                         action_configured = True
                elif action == "Rename Columns":
                    rename_map = {}
                    st.write("Select columns and enter new names:")
                    cols_to_rename_options = sorted(all_cols_wiz)
                    cols_to_rename = st.multiselect("Columns to Rename:", cols_to_rename_options, key=f"rename_select_{action}")
                    for col in cols_to_rename:
                        sanitized_col_key = "".join(c if c.isalnum() else "_" for c in col)
                        new_name = st.text_input(f"New name for '{col}':", value=col, key=f"rename_input_{sanitized_col_key}_{action}")
                        if new_name != col and new_name and new_name.isidentifier():
                            rename_map[col] = new_name
                        elif new_name and not new_name.isidentifier():
                            st.warning(f"'{new_name}' is not a valid Python identifier.")

                    if rename_map and len(rename_map) == len(cols_to_rename):
                        code_string = f"result_data = df.rename(columns={rename_map})"
                        action_configured = True
                    elif cols_to_rename: st.caption("Enter valid new names for all selected columns.")

                elif action == "Handle Missing Data":
                     fill_cols = st.multiselect("Select Columns to Fill NA:", all_cols_wiz, key=f"fillna_cols_{action}")
                     if fill_cols:
                         fill_method = st.radio("Fill Method:", ["Specific Value", "Mean", "Median", "Mode", "Forward Fill (ffill)", "Backward Fill (bfill)", "Drop Rows with NA"], key=f"fillna_method_{action}")
                         code_lines = ["result_data = df.copy()"]
                         valid_op = True
                         if fill_method == "Specific Value":
                             fill_value = st.text_input("Enter Value to Fill NA with:", "0", key=f"fillna_value_{action}")
                             try: fill_value_parsed = float(fill_value)
                             except ValueError: fill_value_parsed = fill_value
                             for col in fill_cols: code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna({repr(fill_value_parsed)})")
                         elif fill_method == "Drop Rows with NA": code_lines.append(f"result_data = result_data.dropna(subset={fill_cols})")
                         else:
                             for col in fill_cols:
                                 if fill_method in ["Mean", "Median"] and col not in numeric_cols_wiz:
                                      st.warning(f"Cannot apply '{fill_method}' to non-numeric column '{col}'. Skipping.")
                                      valid_op = False; break
                                 elif fill_method == "Mean": code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna(pd.to_numeric(result_data['{col}'], errors='coerce').mean())")
                                 elif fill_method == "Median": code_lines.append(f"result_data['{col}'] = result_data['{col}'].fillna(pd.to_numeric(result_data['{col}'], errors='coerce').median())")
                                 elif fill_method == "Mode": code_lines.append(f"col_mode = result_data['{col}'].mode()\nif not col_mode.empty: result_data['{col}'] = result_data['{col}'].fillna(col_mode[0])")
                                 elif fill_method == "Forward Fill (ffill)": code_lines.append(f"result_data['{col}'] = result_data['{col}'].ffill()")
                                 elif fill_method == "Backward Fill (bfill)": code_lines.append(f"result_data['{col}'] = result_data['{col}'].bfill()")

                         if valid_op:
                            code_string = "\n".join(code_lines)
                            action_configured = True

                elif action == "Drop Duplicate Rows":
                    subset_cols = st.multiselect("Consider Columns (Optional, default=all):", all_cols_wiz, key=f"dropdup_subset_{action}")
                    keep_option = st.radio("Keep Which Duplicate:", ['first', 'last', False], format_func=lambda x: str(x) if isinstance(x,str) else "Drop All Duplicates", key=f"dropdup_keep_{action}")
                    code_string = f"result_data = df.drop_duplicates(subset={subset_cols if subset_cols else None}, keep=('{keep_option}' if isinstance('{keep_option}', str) else False) )" # Correct handling of keep=False
                    action_configured = True

                elif action == "Change Data Type":
                    type_col = st.selectbox("Select Column:", all_cols_wiz, key=f"dtype_col_{action}")
                    target_type = st.selectbox("Convert To Type:", ['infer (pandas default)', 'int (nullable)', 'float', 'str', 'datetime', 'category', 'boolean'], key=f"dtype_target_{action}")
                    if type_col and target_type:
                        code_lines = ["result_data = df.copy()"]
                        if target_type == 'infer (pandas default)': code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].convert_dtypes()") # Modern way
                        elif target_type == 'datetime':
                             infer_format = st.checkbox("Try to Infer Datetime Format", value=True, key=f"dtype_infer_{action}")
                             format_str = f", infer_datetime_format={infer_format}" if infer_format else ""
                             code_lines.append(f"result_data['{type_col}'] = pd.to_datetime(result_data['{type_col}'], errors='coerce'{format_str})")
                        elif target_type in ['int (nullable)', 'float']:
                             code_lines.append(f"result_data['{type_col}'] = pd.to_numeric(result_data['{type_col}'], errors='coerce')")
                             if target_type == 'int (nullable)': code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].astype('Int64')")
                        elif target_type == 'boolean':
                            map_dict = {'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False, 't': True, 'f': False, '': pd.NA, 'nan': pd.NA, 'none': pd.NA, 'null': pd.NA} # More comprehensive map including NA values
                            code_lines.append(f"bool_map = {{str(k).lower(): v for k, v in {map_dict}.items()}}")
                            code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].astype(str).str.lower().map(bool_map).astype('boolean')")
                        else: # str, category
                            code_lines.append(f"result_data['{type_col}'] = result_data['{type_col}'].astype('{target_type}')")
                        code_string = "\n".join(code_lines)
                        action_configured = True

                elif action == "String Manipulation (Trim, Case, Replace)":
                    potential_str_cols = categorical_cols_wiz + [c for c in all_cols_wiz if c not in numeric_cols_wiz and c not in datetime_cols_wiz]
                    str_col = st.selectbox("Select Column:", potential_str_cols, key=f"strman_col_{action}")
                    if str_col:
                        str_op = st.radio("Operation:", ["Trim Whitespace", "To Uppercase", "To Lowercase", "To Title Case", "Replace Text"], key=f"strman_op_{action}")
                        code_lines = ["result_data = df.copy()"]
                        base_str_col = f"result_data['{str_col}'].astype(str)" # Ensure string
                        if str_op == "Trim Whitespace": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.strip()")
                        elif str_op == "To Uppercase": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.upper()")
                        elif str_op == "To Lowercase": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.lower()")
                        elif str_op == "To Title Case": code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.title()")
                        elif str_op == "Replace Text":
                            text_to_find = st.text_input("Text to Find:", key=f"strman_find_{action}")
                            text_to_replace = st.text_input("Replace With:", key=f"strman_replace_{action}")
                            use_regex = st.checkbox("Use Regular Expression for 'Text to Find'", key=f"strman_regex_{action}")
                            code_lines.append(f"result_data['{str_col}'] = {base_str_col}.str.replace({repr(text_to_find)}, {repr(text_to_replace)}, regex={use_regex})")
                            action_configured = True # Configured once col/op selected
                        else: action_configured = True # Other ops always configured
                        if action_configured: code_string = "\n".join(code_lines)

                elif action == "Extract from Text (Regex)":
                    extract_col = st.selectbox("Select Column to Extract From:", all_cols_wiz, key=f"extract_col_{action}")
                    regex_pattern = st.text_input("Enter Regex Pattern (with capture groups):", placeholder=r"(\d{4})-(\d{2})-(\d{2})", key=f"extract_regex_{action}")
                    new_col_names_str = st.text_input("New Column Names (comma-separated, optional):", placeholder="Year,Month,Day", key=f"extract_names_{action}")
                    if extract_col and regex_pattern:
                        code_lines = ["result_data = df.copy()"]
                        extract_col_str = f"result_data['{extract_col}'].astype(str)"
                        new_col_names = [name.strip() for name in new_col_names_str.split(',') if name.strip()] if new_col_names_str else None
                        if new_col_names:
                            code_lines.append(f"extracted = {extract_col_str}.str.extract(r{repr(regex_pattern)})")
                            code_lines.append(f"if extracted.shape[1] == len({repr(new_col_names)}): result_data[{repr(new_col_names)}] = extracted")
                            code_lines.append(f"else: print(f'Warning: Regex extracted {{extracted.shape[1]}} groups, but {{len({repr(new_col_names)})}} names provided.')")
                        else:
                            code_lines.append(f"extracted_data = {extract_col_str}.str.extract(r{repr(regex_pattern)})")
                            code_lines.append("result_data = extracted_data")
                        code_string = "\n".join(code_lines)
                        action_configured = True

                elif action == "Date Component Extraction (Year, Month...)":
                     date_col = st.selectbox("Select Column (will attempt Datetime conversion):", datetime_cols_wiz + [c for c in all_cols_wiz if c not in datetime_cols_wiz], key=f"datecomp_col_{action}")
                     component = st.selectbox("Component to Extract:", ["Year", "Month", "Day", "Hour", "Minute", "Second", "Day of Week", "Day Name", "Month Name", "Quarter", "Week of Year"], key=f"datecomp_comp_{action}")
                     if date_col and component:
                         comp_map = {"Year": ".dt.year", "Month": ".dt.month", "Day": ".dt.day", "Hour": ".dt.hour", "Minute": ".dt.minute", "Second": ".dt.second", "Day of Week": ".dt.dayofweek", "Day Name": ".dt.day_name()", "Month Name": ".dt.month_name()", "Quarter": ".dt.quarter", "Week of Year": ".dt.isocalendar().week"}
                         new_date_col_name = f"{date_col}_{component.lower().replace(' ', '_')}"
                         code_lines = [
                             "result_data = df.copy()",
                             f"temp_date_col = pd.to_datetime(result_data['{date_col}'], errors='coerce')",
                             f"result_data['{new_date_col_name}'] = np.nan",
                             f"valid_dates = temp_date_col.notna()",
                             f"if valid_dates.any():",
                             f"    extracted_component = temp_date_col[valid_dates]{comp_map[component]}",
                             f"    result_data.loc[valid_dates, '{new_date_col_name}'] = extracted_component",
                             f"    # Convert integer-like results to nullable Int64",
                             f"    if pd.api.types.is_integer_dtype(extracted_component):",
                             f"        result_data['{new_date_col_name}'] = result_data['{new_date_col_name}'].astype('Int64')"
                             ]
                         code_string = "\n".join(code_lines)
                         action_configured = True

                # === Data Transformation ===
                elif action == "Filter Data":
                    filter_col = st.selectbox("Select Column to Filter:", all_cols_wiz, key=f"filter_col_{action}")
                    if filter_col:
                        col_dtype = df_wizard[filter_col].dtype
                        col_is_numeric = pd.api.types.is_numeric_dtype(col_dtype)
                        col_is_datetime = pd.api.types.is_datetime64_any_dtype(col_dtype) or pd.api.types.is_timedelta64_dtype(col_dtype)
                        col_is_boolean = pd.api.types.is_bool_dtype(col_dtype) or pd.api.types.is_bool(col_dtype) # Check both numpy bool and pandas boolean

                        operators_num = ['==', '!=', '>', '<', '>=', '<=', 'is missing', 'is not missing']
                        operators_str = ['==', '!=', 'contains', 'does not contain', 'starts with', 'ends with', 'is in (comma-sep list)', 'is not in (comma-sep list)', 'is missing', 'is not missing']
                        operators_date = ['==', '!=', '>', '<', '>=', '<=', 'is missing', 'is not missing']
                        operators_bool = ['is True', 'is False', 'is missing', 'is not missing']

                        op = None; val = None; code_lines = []

                        if col_is_numeric:
                            op = st.selectbox("Operator:", operators_num, key=f"filter_op_num_{action}")
                            if op not in ['is missing', 'is not missing']:
                                val = st.number_input(f"Enter Value for {filter_col}:", value=0.0, format="%g", key=f"filter_val_num_{filter_col}")
                                code_lines.append(f"filter_series = pd.to_numeric(df['{filter_col}'], errors='coerce') {op} {val}")
                            elif op == 'is missing': code_lines.append(f"filter_series = pd.to_numeric(df['{filter_col}'], errors='coerce').isnull()")
                            else: code_lines.append(f"filter_series = pd.to_numeric(df['{filter_col}'], errors='coerce').notnull()")
                            action_configured = True
                        elif col_is_datetime:
                             op = st.selectbox("Operator:", operators_date, key=f"filter_op_date_{action}")
                             code_lines.append(f"date_series = pd.to_datetime(df['{filter_col}'], errors='coerce')")
                             if op not in ['is missing', 'is not missing']:
                                 val_date = st.date_input(f"Enter Date for {filter_col}:", key=f"filter_val_date_{filter_col}")
                                 val = pd.Timestamp(f"{val_date}")
                                 code_lines.append(f"filter_series = date_series {op} pd.Timestamp('{val}')")
                             elif op == 'is missing': code_lines.append(f"filter_series = date_series.isnull()")
                             else: code_lines.append(f"filter_series = date_series.notnull()")
                             action_configured = True
                        elif col_is_boolean:
                            op = st.selectbox("Operator:", operators_bool, key=f"filter_op_bool_{action}")
                            code_lines.append(f"bool_series = df['{filter_col}'].astype('boolean')")
                            if op == 'is True': code_lines.append(f"filter_series = bool_series == True")
                            elif op == 'is False': code_lines.append(f"filter_series = bool_series == False")
                            elif op == 'is missing': code_lines.append(f"filter_series = bool_series.isnull()")
                            else: code_lines.append(f"filter_series = bool_series.notnull()")
                            action_configured = True
                        else: # Assume string/categorical
                            op = st.selectbox("Operator:", operators_str, key=f"filter_op_str_{action}")
                            code_lines.append(f"str_series = df['{filter_col}'].astype(str)")
                            if op not in ['is missing', 'is not missing']:
                                val_str = st.text_input(f"Enter Text Value(s) for {filter_col}:", key=f"filter_val_str_{filter_col}")
                                if val_str is not None: # Check if input provided (can be empty string)
                                    if op == 'contains': code_lines.append(f"filter_series = str_series.str.contains({repr(val_str)}, case=False, na=False)")
                                    elif op == 'does not contain': code_lines.append(f"filter_series = ~str_series.str.contains({repr(val_str)}, case=False, na=False)")
                                    elif op == 'starts with': code_lines.append(f"filter_series = str_series.str.startswith({repr(val_str)}, na=False)")
                                    elif op == 'ends with': code_lines.append(f"filter_series = str_series.str.endswith({repr(val_str)}, na=False)")
                                    elif op == 'is in (comma-sep list)':
                                        list_vals = [v.strip() for v in val_str.split(',') if v.strip()]
                                        if list_vals: code_lines.append(f"filter_series = str_series.isin({list_vals})")
                                        else: st.caption("Enter comma-separated values."); action_configured=False
                                    elif op == 'is not in (comma-sep list)':
                                        list_vals = [v.strip() for v in val_str.split(',') if v.strip()]
                                        code_lines.append(f"filter_series = ~str_series.isin({list_vals})")
                                    else: code_lines.append(f"filter_series = str_series {op} {repr(val_str)}")
                                    if action_configured is None: action_configured = True # Assume configured if value provided and not list error
                            elif op == 'is missing':
                                code_lines.append(f"filter_series = df['{filter_col}'].isnull() | (str_series == '') | (str_series.str.lower() == 'nan') | (str_series.str.lower() == 'none') | (str_series.str.lower() == 'null')") # More robust check
                                action_configured = True
                            else: # is not missing
                                code_lines.append(f"filter_series = df['{filter_col}'].notnull() & (str_series != '') & (str_series.str.lower() != 'nan') & (str_series.str.lower() != 'none') & (str_series.str.lower() != 'null')")
                                action_configured = True

                        if action_configured:
                             code_lines.append("result_data = df[filter_series]")
                             code_string = "\n".join(code_lines)

                elif action == "Sort Data":
                    sort_cols = st.multiselect("Select Column(s) to Sort By:", all_cols_wiz, key=f"sort_cols_{action}")
                    if sort_cols:
                        sort_orders_bool = []
                        for col in sort_cols:
                            order = st.radio(f"Sort Order for '{col}':", ["Ascending", "Descending"], key=f"sort_{col}_{action}", horizontal=True)
                            sort_orders_bool.append(order == "Ascending")
                        code_string = f"result_data = df.sort_values(by={sort_cols}, ascending={sort_orders_bool})"
                        action_configured = True

                elif action == "Select Columns":
                     select_cols = st.multiselect("Select Columns to Keep:", all_cols_wiz, default=all_cols_wiz, key=f"select_cols_{action}")
                     if select_cols:
                         code_string = f"result_data = df[{select_cols}]"
                         action_configured = True
                     else: st.warning("Select at least one column.")

                elif action == "Create Calculated Column (Basic Arithmetic)":
                     if not numeric_cols_wiz: st.warning("No numeric columns available for calculations.")
                     st.write("Create a new column based on simple math (+, -, *, /).")
                     new_calc_col_name = st.text_input("New Column Name:", key=f"calc_newname_{action}")
                     col1 = st.selectbox("Select First Numeric Column (or None for constant):", [None] + numeric_cols_wiz, key=f"calc_col1_{action}")
                     op_calc = st.selectbox("Operator:", ['+', '-', '*', '/'], key=f"calc_op_{action}")
                     col2 = st.selectbox("Select Second Numeric Column (or None for constant):", [None] + numeric_cols_wiz, key=f"calc_col2_{action}")
                     constant_val_str = st.text_input("Or Enter Constant Value:", "0", key=f"calc_const_{action}")

                     if new_calc_col_name and new_calc_col_name.isidentifier() and op_calc and (col1 or col2 or constant_val_str):
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
                     elif new_calc_col_name and not new_calc_col_name.isidentifier(): st.warning("Invalid column name.")
                     elif not new_calc_col_name: st.caption("Enter column name.")

                elif action == "Bin Numeric Data (Cut)":
                    if not numeric_cols_wiz: st.warning("No numeric columns available.")
                    bin_col = st.selectbox("Select Numeric Column to Bin:", numeric_cols_wiz, key=f"bin_col_{action}")
                    if bin_col:
                         bin_method = st.radio("Method:", ["Equal Width", "Quantiles", "Custom Edges"], key=f"bin_method_{action}")
                         new_bin_col_name = st.text_input("New Column Name:", f"{bin_col}_binned", key=f"bin_newname_{action}")
                         if new_bin_col_name and new_bin_col_name.isidentifier():
                             code_lines = ["result_data = df.copy()", f"numeric_col = pd.to_numeric(result_data['{bin_col}'], errors='coerce')"]
                             labels_param = "labels=False" # Default to numeric labels

                             if bin_method == "Equal Width":
                                 num_bins = st.slider("Number of Bins:", 2, 50, 5, key=f"bin_num_eq_{action}")
                                 code_lines.append(f"result_data['{new_bin_col_name}'] = pd.cut(numeric_col, bins={num_bins}, {labels_param}, include_lowest=True, duplicates='drop')")
                                 action_configured = True
                             elif bin_method == "Quantiles":
                                 num_q_bins = st.slider("Number of Quantile Bins:", 2, 10, 4, key=f"bin_num_q_{action}")
                                 code_lines.append(f"result_data['{new_bin_col_name}'] = pd.qcut(numeric_col, q={num_q_bins}, {labels_param}, duplicates='drop')")
                                 action_configured = True
                             elif bin_method == "Custom Edges":
                                 edges_str = st.text_input("Edges (comma-separated):", placeholder="0, 10, 50, 100", key=f"bin_edges_{action}")
                                 try:
                                     edges = sorted([float(e.strip()) for e in edges_str.split(',') if e.strip()])
                                     if len(edges) > 1:
                                         code_lines.append(f"bin_edges = {edges}")
                                         code_lines.append(f"result_data['{new_bin_col_name}'] = pd.cut(numeric_col, bins=bin_edges, {labels_param}, include_lowest=True, duplicates='drop')")
                                         action_configured = True
                                     elif edges_str: st.warning("Need >= 2 edges.")
                                 except ValueError: st.error("Invalid edges format.")
                             if action_configured: code_string = "\n".join(code_lines)
                         elif new_bin_col_name and not new_bin_col_name.isidentifier(): st.warning("Invalid column name.")

                elif action == "One-Hot Encode Categorical Column":
                     if not categorical_cols_wiz: st.warning("No categorical columns identified.")
                     ohe_col = st.selectbox("Select Categorical Column:", categorical_cols_wiz, key=f"ohe_col_{action}")
                     if ohe_col:
                         drop_first = st.checkbox("Drop First Category", value=False, key=f"ohe_drop_{action}")
                         code_string = f"result_data = pd.get_dummies(df, columns=['{ohe_col}'], prefix='{ohe_col}', drop_first={drop_first}, dummy_na=False)"
                         action_configured = True

                elif action == "Pivot (Simple)":
                     st.info("Creates a pivot table summary.")
                     pivot_index = st.selectbox("Index (Rows):", all_cols_wiz, key=f"pivot_idx_{action}")
                     pivot_cols = st.selectbox("Columns:", all_cols_wiz, key=f"pivot_cols_{action}")
                     pivot_vals = st.selectbox("Values (Numeric):", numeric_cols_wiz, key=f"pivot_vals_{action}")
                     pivot_agg = st.selectbox("Aggregation:", ['mean', 'sum', 'count', 'median', 'min', 'max'], key=f"pivot_agg_{action}")
                     if pivot_index and pivot_cols and pivot_vals and pivot_agg:
                         if pivot_index == pivot_cols: st.warning("Index and Columns cannot be same.")
                         else:
                              code_string = f"pivot_df = df.copy()\npivot_df['{pivot_vals}'] = pd.to_numeric(pivot_df['{pivot_vals}'], errors='coerce')\nresult_data = pd.pivot_table(pivot_df, index='{pivot_index}', columns='{pivot_cols}', values='{pivot_vals}', aggfunc='{pivot_agg}').reset_index()"
                              action_configured = True

                elif action == "Melt (Unpivot)":
                     st.info("Unpivots DataFrame from wide to long format.")
                     id_vars = st.multiselect("Identifier Variables (Keep):", all_cols_wiz, key=f"melt_idvars_{action}")
                     default_value_vars = [c for c in all_cols_wiz if c not in id_vars]
                     value_vars = st.multiselect("Value Variables (Unpivot):", default_value_vars, default=default_value_vars, key=f"melt_valvars_{action}")
                     var_name = st.text_input("New Variable Column Name:", "Variable", key=f"melt_varname_{action}")
                     value_name = st.text_input("New Value Column Name:", "Value", key=f"melt_valuename_{action}")
                     if id_vars and var_name and value_name and var_name.isidentifier() and value_name.isidentifier():
                         value_vars_param = f"value_vars={value_vars}" if value_vars else ""
                         code_string = f"result_data = pd.melt(df, id_vars={id_vars}, {value_vars_param}, var_name='{var_name}', value_name='{value_name}')"
                         action_configured = True
                     elif not id_vars: st.warning("Select Identifier Variable(s).")
                     elif not (var_name and value_name and var_name.isidentifier() and value_name.isidentifier()): st.warning("Enter valid new column names.")

                # === Aggregation & Analysis ===
                elif action == "Calculate Single Aggregation (Sum, Mean...)":
                    if not numeric_cols_wiz: st.warning("No numeric columns.")
                    agg_col = st.selectbox("Select Numeric Column:", numeric_cols_wiz, key=f"sagg_col_{action}")
                    agg_func = st.selectbox("Aggregation Function:", ['sum', 'mean', 'median', 'min', 'max', 'count', 'nunique', 'std', 'var'], key=f"sagg_func_{action}")
                    if agg_col and agg_func:
                        code_string = f"agg_numeric_col = pd.to_numeric(df['{agg_col}'], errors='coerce')\nresult_data = agg_numeric_col.{agg_func}()"
                        action_configured = True

                elif action == "Group By & Aggregate":
                    group_cols = st.multiselect("Group By Columns:", categorical_cols_wiz + datetime_cols_wiz, key=f"gagg_groupcols_{action}")
                    st.write("Define Aggregations:")
                    num_aggs = st.number_input("Number of Aggregations:", min_value=1, value=1, key=f"gagg_numaggs_{action}")
                    named_aggs_list = []; valid_agg_config = True; cols_to_numeric_check = set()
                    for i in range(num_aggs):
                        st.markdown(f"**Aggregation #{i+1}**")
                        agg_col_group = st.selectbox(f"Aggregate Column:", all_cols_wiz, key=f"gagg_aggcol_{i}_{action}")
                        agg_func_group = st.selectbox(f"Function:", ['sum', 'mean', 'median', 'min', 'max', 'count', 'nunique', 'std', 'var', 'first', 'last'], key=f"gagg_aggfunc_{i}_{action}")
                        default_name = f"{agg_col_group}_{agg_func_group}".replace('[^A-Za-z0-9_]+', '') if agg_col_group else f"agg_{i+1}"
                        new_agg_name = st.text_input(f"Result Column Name:", value=default_name, key=f"gagg_aggname_{i}_{action}")
                        if not agg_col_group: st.warning(f"#{i+1}: Select column."); valid_agg_config = False
                        elif not new_agg_name or not new_agg_name.isidentifier(): st.warning(f"#{i+1}: Enter valid name."); valid_agg_config = False
                        else:
                            if agg_func_group in ['sum', 'mean', 'median', 'std', 'var']: cols_to_numeric_check.add(agg_col_group)
                            named_aggs_list.append(f"{new_agg_name}=pd.NamedAgg(column='{agg_col_group}', aggfunc='{agg_func_group}')")
                    if group_cols and named_aggs_list and valid_agg_config:
                         numeric_checks_code = "\n".join([f"df_agg['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')" for col in cols_to_numeric_check])
                         named_aggs_str = ", ".join(named_aggs_list)
                         code_string = f"df_agg = df.copy()\n{numeric_checks_code}\nresult_data = df_agg.groupby({group_cols}).agg({named_aggs_str}).reset_index()"
                         action_configured = True
                    elif not group_cols: st.warning("Select Group By column(s).")

                elif action == "Calculate Rolling Window Statistics":
                    if not numeric_cols_wiz: st.warning("No numeric columns.")
                    roll_col = st.selectbox("Select Numeric Column:", numeric_cols_wiz, key=f"roll_col_{action}")
                    if roll_col:
                        window_size = st.number_input("Window Size:", min_value=2, value=3, key=f"roll_window_{action}")
                        roll_func = st.selectbox("Function:", ['mean', 'sum', 'median', 'std', 'min', 'max', 'var'], key=f"roll_func_{action}")
                        center = st.checkbox("Center Window?", value=False, key=f"roll_center_{action}")
                        min_periods = st.number_input("Min Periods (Optional):", min_value=1, value=None, placeholder="Default", key=f"roll_minp_{action}")
                        sort_by_col = st.selectbox("Sort by (Optional):", [None] + all_cols_wiz, key=f"roll_sort_{action}")
                        new_roll_col_name = st.text_input("New Column Name:", f"{roll_col}_roll_{roll_func}_{window_size}", key=f"roll_newname_{action}")
                        if new_roll_col_name and new_roll_col_name.isidentifier():
                             code_lines = ["result_data = df.copy()"]
                             if sort_by_col: code_lines.append(f"result_data = result_data.sort_values(by='{sort_by_col}')")
                             code_lines.append(f"rolling_col_numeric = pd.to_numeric(result_data['{roll_col}'], errors='coerce')")
                             min_periods_param = f", min_periods={min_periods}" if min_periods is not None else ""
                             code_lines.append(f"result_data['{new_roll_col_name}'] = rolling_col_numeric.rolling(window={window_size}, center={center}{min_periods_param}).{roll_func}()")
                             if sort_by_col: code_lines.append(f"# Data remains sorted by '{sort_by_col}'")
                             code_string = "\n".join(code_lines); action_configured = True
                        elif new_roll_col_name and not new_roll_col_name.isidentifier(): st.warning("Invalid name.")

                elif action == "Calculate Cumulative Statistics":
                     if not numeric_cols_wiz: st.warning("No numeric columns.")
                     cum_col = st.selectbox("Select Numeric Column:", numeric_cols_wiz, key=f"cum_col_{action}")
                     if cum_col:
                         cum_func = st.selectbox("Function:", ['sum', 'prod', 'min', 'max'], key=f"cum_func_{action}")
                         sort_by_col_cum = st.selectbox("Sort by (Optional):", [None] + all_cols_wiz, key=f"cum_sort_{action}")
                         new_cum_col_name = st.text_input("New Column Name:", f"{cum_col}_cum_{cum_func}", key=f"cum_newname_{action}")
                         if new_cum_col_name and new_cum_col_name.isidentifier():
                             code_lines = ["result_data = df.copy()"]
                             if sort_by_col_cum: code_lines.append(f"result_data = result_data.sort_values(by='{sort_by_col_cum}')")
                             code_lines.append(f"cum_col_numeric = pd.to_numeric(result_data['{cum_col}'], errors='coerce')")
                             code_lines.append(f"result_data['{new_cum_col_name}'] = cum_col_numeric.cum{cum_func}()")
                             if sort_by_col_cum: code_lines.append(f"# Data remains sorted by '{sort_by_col_cum}'")
                             code_string = "\n".join(code_lines); action_configured = True
                         elif new_cum_col_name and not new_cum_col_name.isidentifier(): st.warning("Invalid name.")

                elif action == "Rank Data":
                     if not numeric_cols_wiz: st.warning("No numeric columns.")
                     rank_col = st.selectbox("Select Numeric Column to Rank:", numeric_cols_wiz, key=f"rank_col_{action}")
                     if rank_col:
                         rank_ascending = st.radio("Order:", ["Ascending", "Descending"], key=f"rank_order_{action}", horizontal=True) == "Ascending"
                         rank_method = st.selectbox("Ties Method:", ['average', 'min', 'max', 'first', 'dense'], key=f"rank_method_{action}")
                         new_rank_col_name = st.text_input("New Rank Column Name:", f"{rank_col}_rank", key=f"rank_newname_{action}")
                         if new_rank_col_name and new_rank_col_name.isidentifier():
                             code_string = f"result_data = df.copy()\nrank_col_numeric = pd.to_numeric(result_data['{rank_col}'], errors='coerce')\nresult_data['{new_rank_col_name}'] = rank_col_numeric.rank(method='{rank_method}', ascending={rank_ascending}, na_option='bottom')" # Keep NaNs at bottom
                             action_configured = True
                         elif new_rank_col_name and not new_rank_col_name.isidentifier(): st.warning("Invalid name.")

                # === Visualization ===
                elif action == "Plot Histogram (Numeric)":
                    if not numeric_cols_wiz: st.warning("No numeric columns.")
                    hist_col = st.selectbox("Select Numeric Column:", numeric_cols_wiz, key=f"hist_col_{action}")
                    if hist_col:
                        bins = st.slider("Bins:", 5, 100, 20, key=f"hist_bins_{action}")
                        kde = st.checkbox("Density Curve (KDE)", value=True, key=f"hist_kde_{action}")
                        code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nplot_col_numeric = pd.to_numeric(df['{hist_col}'], errors='coerce')\nsns.histplot(x=plot_col_numeric.dropna(), bins={bins}, kde={kde}, ax=ax)\nax.set_title('Histogram of {hist_col}')\nplt.tight_layout()\nresult_data = fig"""
                        action_configured = True
                elif action == "Plot Density Plot (Numeric)":
                    if not numeric_cols_wiz: st.warning("No numeric columns.")
                    dens_col = st.selectbox("Select Numeric Column:", numeric_cols_wiz, key=f"dens_col_{action}")
                    if dens_col:
                        shade = st.checkbox("Shade Area", value=True, key=f"dens_shade_{action}")
                        code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nplot_col_numeric = pd.to_numeric(df['{dens_col}'], errors='coerce')\nsns.kdeplot(x=plot_col_numeric.dropna(), fill={shade}, ax=ax)\nax.set_title('Density Plot of {dens_col}')\nplt.tight_layout()\nresult_data = fig"""
                        action_configured = True
                elif action == "Plot Count Plot (Categorical)":
                    if not categorical_cols_wiz: st.warning("No categorical columns.")
                    count_col = st.selectbox("Select Categorical Column:", categorical_cols_wiz, key=f"count_col_{action}")
                    if count_col:
                        top_n_check = st.checkbox("Show Only Top N?", value=True, key=f"count_topn_check_{action}")
                        top_n = 20 ; plot_order_code = f"order=plot_data_count['{count_col}'].value_counts().index"
                        if top_n_check:
                             top_n = st.slider("Top N:", 5, 50, 20, key=f"count_topn_slider_{action}")
                             plot_order_code = f"order=plot_data_count['{count_col}'].value_counts().nlargest({top_n}).index"
                        code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\ncol = '{count_col}'\nif col in df.columns:\n    plot_data_count = df[[col]].dropna()\n    if not plot_data_count.empty:\n        plot_order = {plot_order_code}\n        sns.countplot(y=col, data=plot_data_count, {plot_order_code}, ax=ax, color='skyblue')\n        ax.set_title(f'Counts for {{col}}' + (' (Top {top_n})' if {top_n_check} and df[col].nunique() > {top_n} else ''))\n        plt.tight_layout()\n        result_data = fig\n    else: result_data=None\nelse: result_data=None"""
                        action_configured = True
                elif action == "Plot Bar Chart (Aggregated)":
                     st.info("Use on 'Group By' result or suitable DF.")
                     prev_result = st.session_state.get('current_action_result_display', None)
                     df_options = {"Use Original DataFrame": df_wizard}
                     if isinstance(prev_result, pd.DataFrame): df_options["Use Previous Wizard Result"] = prev_result
                     selected_df_key = st.radio("Data From:", list(df_options.keys()), horizontal=True, key=f"bar_df_source_{action}")
                     bar_df_source = df_options[selected_df_key]
                     if isinstance(bar_df_source, pd.DataFrame) and len(bar_df_source.columns) >= 2:
                         num_cols_bar, cat_cols_bar, _ = get_column_types(bar_df_source)
                         x_col_bar = st.selectbox("X-axis (Categorical):", cat_cols_bar, key=f"bar_x_{action}")
                         y_col_bar = st.selectbox("Y-axis (Numeric):", num_cols_bar, key=f"bar_y_{action}")
                         if x_col_bar and y_col_bar:
                             code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nplot_source_df['{y_col_bar}'] = pd.to_numeric(plot_source_df['{y_col_bar}'], errors='coerce')\nbar_plot_data = plot_source_df.nlargest(30, '{y_col_bar}')\nsns.barplot(x='{x_col_bar}', y='{y_col_bar}', data=bar_plot_data, ax=ax, color='lightcoral')\nax.set_title('Bar Chart: {y_col_bar} by {x_col_bar}')\nplt.xticks(rotation=60, ha='right')\nplt.tight_layout()\nresult_data = fig"""
                             action_configured = True # Need to pass bar_df_source later
                         elif not x_col_bar or not y_col_bar: st.caption("Select columns.")
                     else: st.warning("Selected data not suitable.")
                elif action == "Plot Line Chart":
                     st.info("Select X (often Date/Time/Seq) and Y (Numeric).")
                     x_col_line = st.selectbox("X-axis Column:", all_cols_wiz, key=f"line_x_{action}")
                     y_col_line = st.selectbox("Y-axis Column (Numeric):", numeric_cols_wiz, key=f"line_y_{action}")
                     hue_col_line = st.selectbox("Group Lines By (Optional):", [None] + categorical_cols_wiz, key=f"line_hue_{action}")
                     if x_col_line and y_col_line:
                         hue_param = f", hue='{hue_col_line}'" if hue_col_line else ""
                         code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nplot_df_line = df.copy()\nplot_df_line['{y_col_line}'] = pd.to_numeric(plot_df_line['{y_col_line}'], errors='coerce')\nx_col = '{x_col_line}'\ntry:\n    if pd.api.types.is_datetime64_any_dtype(plot_df_line[x_col]):\n        plot_df_line[x_col] = pd.to_datetime(plot_df_line[x_col], errors='coerce')\n    plot_df_line.dropna(subset=[x_col, '{y_col_line}'], inplace=True)\n    if not plot_df_line.empty: plot_df_line = plot_df_line.sort_values(by=x_col)\nexcept Exception: pass\nif not plot_df_line.empty:\n    sns.lineplot(x=x_col, y='{y_col_line}'{hue_param}, data=plot_df_line, ax=ax, marker='o', markersize=4)\n    ax.set_title('Line Chart: {y_col_line} over {x_col}')\n    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); result_data = fig\nelse: result_data = None"""
                         action_configured = True
                elif action == "Plot Scatter Plot":
                     if len(numeric_cols_wiz) < 2: st.warning("Need >= 2 numeric columns.")
                     st.info("Visualize relationship between two numeric variables.")
                     x_col_scatter = st.selectbox("X-axis (Numeric):", numeric_cols_wiz, key=f"scatter_x_{action}")
                     y_col_scatter = st.selectbox("Y-axis (Numeric):", numeric_cols_wiz, key=f"scatter_y_{action}")
                     hue_col_scatter = st.selectbox("Color By (Optional):", [None] + categorical_cols_wiz, key=f"scatter_hue_{action}")
                     size_col_scatter = st.selectbox("Size By (Optional):", [None] + numeric_cols_wiz, key=f"scatter_size_{action}")
                     if x_col_scatter and y_col_scatter:
                         hue_param = f", hue='{hue_col_scatter}'" if hue_col_scatter else ""
                         size_param = f", size='{size_col_scatter}'" if size_col_scatter else ""
                         code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nscatter_df = df.copy()\nscatter_df['{x_col_scatter}'] = pd.to_numeric(scatter_df['{x_col_scatter}'], errors='coerce')\nscatter_df['{y_col_scatter}'] = pd.to_numeric(scatter_df['{y_col_scatter}'], errors='coerce')\nif '{size_col_scatter}' != 'None': scatter_df['{size_col_scatter}'] = pd.to_numeric(scatter_df['{size_col_scatter}'], errors='coerce')\nscatter_df.dropna(subset=['{x_col_scatter}', '{y_col_scatter}'{(", '" + size_col_scatter + "'") if size_col_scatter else ""}], inplace=True)\nsns.scatterplot(x='{x_col_scatter}', y='{y_col_scatter}'{hue_param}{size_param}, data=scatter_df, ax=ax, alpha=0.6)\nax.set_title('Scatter Plot: {y_col_scatter} vs {x_col_scatter}')\nplt.tight_layout()\nresult_data = fig"""
                         action_configured = True
                elif action == "Plot Box Plot (Numeric vs Cat)":
                     if not numeric_cols_wiz or not categorical_cols_wiz: st.warning("Need numeric and categorical columns.")
                     st.info("Compare numeric distribution across categories.")
                     x_col_box = st.selectbox("Categorical Column (X-axis):", categorical_cols_wiz, key=f"box_x_{action}")
                     y_col_box = st.selectbox("Numeric Column (Y-axis):", numeric_cols_wiz, key=f"box_y_{action}")
                     if x_col_box and y_col_box:
                         limit_cats_box = st.checkbox("Limit Categories?", value=True, key=f"box_limit_{action}")
                         top_n_box = 15
                         if limit_cats_box: top_n_box = st.slider("Max Categories:", 5, 50, 15, key=f"box_topn_{action}")
                         code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\nfig, ax = plt.subplots()\nlimit_cats = {limit_cats_box}\ntop_n = {top_n_box}\nx_col = '{x_col_box}'\ny_col = '{y_col_box}'\nplot_data_box = df.copy()\nplot_data_box[y_col] = pd.to_numeric(plot_data_box[y_col], errors='coerce')\nplot_data_box.dropna(subset=[x_col, y_col], inplace=True)\nif plot_data_box.empty: result_data=None\nelse:\n    if limit_cats and plot_data_box[x_col].nunique() > top_n:\n        plot_order_box = plot_data_box[x_col].value_counts().nlargest(top_n).index\n        plot_data_box = plot_data_box[plot_data_box[x_col].isin(plot_order_box)]\n        sns.boxplot(x=x_col, y=y_col, data=plot_data_box, order=plot_order_box, ax=ax, showfliers=False)\n        ax.set_title(f'Box Plot: {{y_col}} by Top {{top_n}} {{x_col}}')\n    else:\n        plot_order_box = plot_data_box[x_col].value_counts().index\n        sns.boxplot(x=x_col, y=y_col, data=plot_data_box, order=plot_order_box, ax=ax, showfliers=False)\n        ax.set_title(f'Box Plot: {{y_col}} by {{x_col}}')\n    plt.xticks(rotation=60, ha='right')\n    plt.tight_layout()\n    result_data = fig"""
                         action_configured = True
                elif action == "Plot Correlation Heatmap (Numeric)":
                     if len(numeric_cols_wiz) < 2: st.warning("Need >= 2 numeric columns.")
                     st.info("Shows correlation between numeric columns.")
                     default_corr_cols = numeric_cols_wiz[:min(len(numeric_cols_wiz), 15)]
                     corr_cols = st.multiselect("Select Numeric Columns:", numeric_cols_wiz, default=default_corr_cols, key=f"corr_cols_{action}")
                     if len(corr_cols) >= 2:
                         corr_method = st.selectbox("Method:", ['pearson', 'kendall', 'spearman'], key=f"corr_method_{action}")
                         code_string = f"""import matplotlib.pyplot as plt\nimport seaborn as sns\ncorr_df = df[{corr_cols}].copy()\nfor col in {corr_cols}: corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')\ncorr_matrix = corr_df.corr(method='{corr_method}')\nfig, ax = plt.subplots(figsize=(max(6, len({corr_cols})*0.7), max(5, len({corr_cols})*0.6)))\nsns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax, annot_kws={{"size": 8}})\nax.set_title('Correlation Heatmap ({corr_method.capitalize()})')\nplt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()\nresult_data = fig"""
                         action_configured = True
                     else: st.warning("Select >= 2 columns.")

            except Exception as e:
                st.error(f"Error configuring action '{action}': {e}")
                st.error(traceback.format_exc())
                code_string = "# Error during action configuration"
                action_configured = False


            # --- Trigger Wizard Execution ---
            st.subheader("3. Apply Wizard Action")
            apply_col1_wiz, apply_col2_wiz = st.columns([1, 3])
            with apply_col1_wiz:
                apply_button_pressed_wiz = st.button(f"Apply: {action}", key=f"apply_{action}", use_container_width=True, disabled=not action_configured)

            if not action_configured and action and not action.startswith("---"):
                 with apply_col2_wiz: st.caption("👈 Configure all options first.")

            if apply_button_pressed_wiz and action_configured:
                if code_string and code_string != "# Select an action and configure options" and not code_string.startswith("# Error"):
                    st.session_state.generated_code = code_string
                    st.session_state.result_type = None
                    st.session_state.show_profile_report = False ; st.session_state.show_sweetviz_report = False
                    st.session_state.current_action_result_display = None
                    st.session_state.ml_evaluation = {} ; st.session_state.ml_pipeline = None
                    report_placeholder.empty()

                    with st.spinner(f"Applying wizard action: {action}..."):
                        local_vars = {'df': df_wizard, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'result_data': None, 'io': io} # Add io for buffer
                        if action == "Plot Bar Chart (Aggregated)" and isinstance(bar_df_source, pd.DataFrame):
                            local_vars['plot_source_df'] = bar_df_source

                        try:
                            exec(code_string, {'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'io': io}, local_vars) # Provide imports to exec context
                            res = local_vars.get('result_data')
                            current_result_to_display = None; current_result_type = None

                            if isinstance(res, pd.DataFrame): current_result_type = 'dataframe'; current_result_to_display = res
                            elif isinstance(res, pd.Series): current_result_type = 'dataframe'; current_result_to_display = res.to_frame()
                            elif isinstance(res, plt.Figure): current_result_type = 'plot'; current_result_to_display = res
                            elif isinstance(res, str) and action == "Show Columns & Types": current_result_type = 'text_info'; current_result_to_display = res # Special case for df.info() string
                            elif res is not None : current_result_type = 'scalar_text'; current_result_to_display = res
                            else: current_result_type = None; current_result_to_display = None

                            st.session_state.current_action_result_display = current_result_to_display
                            st.session_state.result_type = current_result_type

                            if current_result_to_display is not None: st.success("Action applied successfully!")
                            else: st.warning("Action ran, but no displayable result produced.")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error executing action: {e}")
                            st.error(traceback.format_exc())
                            st.session_state.current_action_result_display = None
                            with st.expander("Show Failing Code"): st.code(st.session_state.generated_code, language='python')
                else:
                     with apply_col2_wiz: st.warning("Could not apply action. Check configuration.")


            # --- Display Wizard Results ---
            st.subheader("📊 Wizard Results")
            result_display_placeholder_wiz = st.empty()
            with result_display_placeholder_wiz.container():
                current_result = st.session_state.get('current_action_result_display', None)
                current_type = st.session_state.get('result_type', None)

                if current_result is not None:
                    try:
                        if current_type == 'dataframe':
                            st.dataframe(current_result)
                            res_df = current_result
                            col1_dl, col2_dl = st.columns(2)
                            with col1_dl: st.download_button("Download CSV", convert_df_to_csv(res_df), "wizard_result.csv", "text/csv", use_container_width=True, key="dl_csv")
                            with col2_dl: st.download_button("Download Excel", convert_df_to_excel(res_df), "wizard_result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_excel")
                        elif current_type == 'plot':
                            if isinstance(current_result, plt.Figure):
                                 st.pyplot(current_result)
                                 # Need to manage the figure object carefully for download after display
                                 # Clone figure for saving bytes? Or save before closing? Let's try saving before closing.
                                 plot_bytes = save_plot_to_bytes(current_result) # This function now closes the figure
                                 if plot_bytes: st.download_button("Download Plot PNG", plot_bytes, "wizard_plot.png", "image/png", use_container_width=True, key="dl_plot")
                                 else: st.warning("Could not generate plot download.")
                            else: st.error("Result is plot type, but not a valid Figure.")
                        elif current_type == 'scalar_text':
                             st.write("Result:")
                             if isinstance(current_result, (dict, list)): st.json(current_result)
                             else: st.write(current_result)
                             try:
                                  scalar_data = str(current_result).encode('utf-8'); st.download_button("Download Text", scalar_data, "wizard_result.txt", "text/plain", use_container_width=True, key="dl_scalar")
                             except Exception as scalar_dl_err: st.warning(f"Could not generate text download: {scalar_dl_err}")
                        elif current_type == 'text_info': # Handle df.info() string
                             st.text(current_result) # Use st.text for preformatted
                             st.download_button("Download Info", current_result.encode('utf-8'), "df_info.txt", "text/plain", use_container_width=True, key="dl_info")

                        if st.session_state.generated_code:
                             with st.expander("Show Python Code for This Result"): st.code(st.session_state.generated_code, language='python')
                    except Exception as display_err:
                         st.error(f"Error displaying wizard result: {display_err}") ; st.error(traceback.format_exc())
                else:
                     if not st.session_state.get('show_profile_report', False) and not st.session_state.get('show_sweetviz_report', False):
                         st.caption("Apply an action from the wizard above to see results here.")


    # --- Tab 3: Machine Learning ---
    with tab3:
        st.header("🤖 Machine Learning Workbench")
        st.markdown("Build and evaluate basic supervised learning models.")

        df_ml = st.session_state.df
        if df_ml is None or df_ml.empty:
            st.warning("Load data first to use the ML Workbench.")
        else:
            all_cols_ml = df_ml.columns.tolist()
            num_cols_ml, cat_cols_ml, dt_cols_ml = get_column_types(df_ml)

            st.subheader("1. Define ML Problem")
            ml_prob_options = ["Regression (Predict a Number)", "Classification (Predict a Category)"]
            ml_prob_index = 0 if st.session_state.ml_problem_type == "Regression" else 1 if st.session_state.ml_problem_type == "Classification" else None
            problem_type = st.radio("Problem Type:", ml_prob_options, index=ml_prob_index, key="ml_problem_type_radio", horizontal=True)

            if problem_type != st.session_state.ml_problem_type:
                st.session_state.ml_problem_type = problem_type
                st.session_state.ml_features = [] ; st.session_state.ml_target = None ; st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {} ; st.session_state.ml_code = "" ; st.session_state.ml_model_name=None
                st.rerun()

            st.subheader("2. Select Features (X) and Target (y)")
            if st.session_state.ml_problem_type:
                target_col_options = [None]
                if st.session_state.ml_problem_type == "Regression": target_col_options.extend(num_cols_ml) ; target_help = "Select numeric target."
                else: target_col_options.extend(cat_cols_ml + [c for c in all_cols_ml if df_ml[c].dtype=='boolean' or pd.api.types.is_integer_dtype(df_ml[c])]) ; target_help = "Select categorical/boolean/integer target." # Allow int targets too

                current_target = st.session_state.get('ml_target')
                target_index = target_col_options.index(current_target) if current_target in target_col_options else 0
                target = st.selectbox("Target Column (y):", options=target_col_options, index=target_index, key="ml_target_select", help=target_help)

                if target != current_target:
                    st.session_state.ml_target = target
                    st.session_state.ml_features = [] ; st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {} ; st.session_state.ml_code = "" ; st.session_state.ml_model_name=None
                    st.rerun()

                if st.session_state.ml_target:
                    available_features = [col for col in all_cols_ml if col != st.session_state.ml_target]
                    # Default features: all numeric + categorical (excluding target and datetime)
                    default_features_ml = [f for f in num_cols_ml + cat_cols_ml if f != st.session_state.ml_target and f not in dt_cols_ml]
                    current_features = st.session_state.get('ml_features', [])
                    features = st.multiselect( "Feature Columns (X):", options=available_features, default=current_features if current_features else default_features_ml, key="ml_features_select", help="Choose columns for prediction.")
                    if features != current_features:
                         st.session_state.ml_features = features
                         st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {} ; st.session_state.ml_code = "" ; st.session_state.ml_model_name=None
                         # No rerun needed here

                else: st.caption("Select target column.")

                if st.session_state.ml_features and st.session_state.ml_target:
                    st.subheader("3. Configure Preprocessing & Model")
                    # Identify feature types *within the selected features*
                    selected_features_df = df_ml[st.session_state.ml_features]
                    numeric_features_selected = selected_features_df.select_dtypes(include=np.number).columns.tolist()
                    # Identify categorical more carefully - include object, category but exclude high cardinality?
                    potential_cat_features = selected_features_df.select_dtypes(include=['object', 'category']).columns.tolist()
                    categorical_features_selected = []
                    for col in potential_cat_features:
                         # Simple cardinality check
                         if df_ml[col].nunique(dropna=False) < 50: # Example threshold
                             categorical_features_selected.append(col)
                         else: st.caption(f"Note: Feature '{col}' has high unique values, excluded from categorical encoding.")

                    col1_ml_config, col2_ml_config = st.columns(2)
                    with col1_ml_config:
                        st.markdown("**Preprocessing**")
                        num_imputer_strategy = st.selectbox("Numeric Imputation:", ['mean', 'median'], key="ml_num_impute")
                        cat_imputer_strategy = st.selectbox("Categorical Imputation:", ['most_frequent', 'constant'], key="ml_cat_impute")
                        cat_fill_value = "missing" if cat_imputer_strategy == 'constant' else None
                        scaler_option = st.selectbox("Numeric Scaling:", ["StandardScaler", "MinMaxScaler", "None"], key="ml_scaler")
                        encoder_option = st.selectbox("Categorical Encoding:", ["OneHotEncoder"], key="ml_encoder")

                    with col2_ml_config:
                        st.markdown("**Model**")
                        model_options = []
                        if st.session_state.ml_problem_type == "Regression": model_options = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"]
                        else: model_options = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Classifier (Linear)", "K-Nearest Neighbors"]
                        current_model = st.session_state.get('ml_model_name')
                        model_index = model_options.index(current_model) if current_model in model_options else 0
                        model_name = st.selectbox("Select Model:", model_options, index=model_index, key="ml_model_select")
                        st.session_state.ml_model_name = model_name

                    st.subheader("4. Train Model & Evaluate")
                    test_size = st.slider("Test Set Size (%):", 10, 50, 25, 5, key="ml_test_split") / 100.0

                    if st.button("🚀 Train and Evaluate Model", key="ml_train_button", use_container_width=True):
                        ml_code_lines = ["# --- ML Workflow Code ---"] # Start generating code
                        # --- [Append all imports and code generation steps as in previous response] ---
                        # ... (imports based on selected options) ...
                        ml_code_lines.append("import pandas as pd"); ml_code_lines.append("from sklearn.model_selection import train_test_split")
                        ml_code_lines.append("from sklearn.impute import SimpleImputer"); ml_code_lines.append("from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder")
                        ml_code_lines.append("from sklearn.compose import ColumnTransformer"); ml_code_lines.append("from sklearn.pipeline import Pipeline")
                        # Model imports...
                        if model_name == "Linear Regression": ml_code_lines.append("from sklearn.linear_model import LinearRegression")
                        elif model_name == "Ridge Regression": ml_code_lines.append("from sklearn.linear_model import Ridge")
                        elif model_name == "Lasso Regression": ml_code_lines.append("from sklearn.linear_model import Lasso")
                        elif model_name == "Logistic Regression": ml_code_lines.append("from sklearn.linear_model import LogisticRegression")
                        elif model_name == "Decision Tree Regressor": ml_code_lines.append("from sklearn.tree import DecisionTreeRegressor")
                        elif model_name == "Decision Tree Classifier": ml_code_lines.append("from sklearn.tree import DecisionTreeClassifier")
                        elif model_name == "Random Forest Regressor": ml_code_lines.append("from sklearn.ensemble import RandomForestRegressor")
                        elif model_name == "Random Forest Classifier": ml_code_lines.append("from sklearn.ensemble import RandomForestClassifier")
                        elif model_name == "Support Vector Classifier (Linear)": ml_code_lines.append("from sklearn.svm import SVC")
                        elif model_name == "K-Nearest Neighbors": ml_code_lines.append("from sklearn.neighbors import KNeighborsClassifier")
                        # Metric imports...
                        if st.session_state.ml_problem_type == "Regression": ml_code_lines.append("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score")
                        else: ml_code_lines.append("from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay"); ml_code_lines.append("import matplotlib.pyplot as plt")
                        # Data prep...
                        ml_code_lines.append("\n# --- Data Preparation ---")
                        ml_code_lines.append(f"target_col = '{st.session_state.ml_target}'")
                        ml_code_lines.append(f"feature_cols = {st.session_state.ml_features}")
                        ml_code_lines.append(f"df_ml = df[feature_cols + [target_col]].copy()")
                        ml_code_lines.append(f"df_ml.dropna(subset=[target_col], inplace=True)")
                        ml_code_lines.append("X = df_ml[feature_cols]")
                        ml_code_lines.append("y = df_ml[target_col]")
                        # Regression target check...
                        if st.session_state.ml_problem_type == "Regression": ml_code_lines.append("y = pd.to_numeric(y, errors='coerce') # Ensure target is numeric") ; ml_code_lines.append("X = X[y.notna()]") ; ml_code_lines.append("y = y[y.notna()] # Drop rows where target conversion failed")
                        # Split...
                        ml_code_lines.append("\n# --- Train/Test Split ---")
                        ml_code_lines.append(f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}, random_state=42, stratify=y if '{st.session_state.ml_problem_type}' == 'Classification' and y.nunique() > 1 else None)") # Stratify for classification if possible
                        # Preprocessing...
                        ml_code_lines.append("\n# --- Preprocessing Pipeline ---")
                        ml_code_lines.append(f"numeric_features = {numeric_features_selected}")
                        ml_code_lines.append(f"categorical_features = {categorical_features_selected}")
                        ml_code_lines.append(f"numeric_steps = [('imputer', SimpleImputer(strategy='{num_imputer_strategy}'))]")
                        if scaler_option == "StandardScaler": ml_code_lines.append(f", ('scaler', StandardScaler())")
                        elif scaler_option == "MinMaxScaler": ml_code_lines.append(f", ('scaler', MinMaxScaler())")
                        ml_code_lines.append("numeric_transformer = Pipeline(steps=numeric_steps)")
                        ml_code_lines.append(f"categorical_steps = [('imputer', SimpleImputer(strategy='{cat_imputer_strategy}', fill_value='{cat_fill_value if cat_fill_value else 'missing'}'))")
                        if encoder_option == "OneHotEncoder": ml_code_lines.append(f", ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))")
                        ml_code_lines.append("]") ; ml_code_lines.append("categorical_transformer = Pipeline(steps=categorical_steps)")
                        ml_code_lines.append("preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough')")
                        # Model...
                        ml_code_lines.append("\n# --- Model Selection ---")
                        if model_name in ["Random Forest Regressor", "Random Forest Classifier", "K-Nearest Neighbors"] : ml_code_lines.append(f"model = {model_name.replace(' ', '')}(random_state=42, n_estimators=50, n_jobs=-1)") # Add n_jobs
                        elif model_name == "Logistic Regression": ml_code_lines.append(f"model = {model_name.replace(' ', '')}(random_state=42, max_iter=200, n_jobs=-1)")
                        elif model_name == "Support Vector Classifier (Linear)": ml_code_lines.append(f"model = SVC(kernel='linear', probability=True, random_state=42)")
                        elif model_name in ["Decision Tree Regressor", "Decision Tree Classifier"]: ml_code_lines.append(f"model = {model_name.replace(' ', '')}(random_state=42)")
                        else: ml_code_lines.append(f"model = {model_name.replace(' ', '')}()")
                        # Pipeline...
                        ml_code_lines.append("\n# --- Full Pipeline ---")
                        ml_code_lines.append(f"pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])")
                        # Training...
                        ml_code_lines.append("\n# --- Training ---"); ml_code_lines.append("pipeline.fit(X_train, y_train)")
                        # Prediction...
                        ml_code_lines.append("\n# --- Prediction ---"); ml_code_lines.append("y_pred = pipeline.predict(X_test)")
                        # Evaluation code...
                        ml_code_lines.append("\n# --- Evaluation ---")
                        if st.session_state.ml_problem_type == "Regression":
                            ml_code_lines.append("mae = mean_absolute_error(y_test, y_pred)"); ml_code_lines.append("mse = mean_squared_error(y_test, y_pred)"); ml_code_lines.append("r2 = r2_score(y_test, y_pred)")
                            ml_code_lines.append("print(f'MAE: {mae:.4f}'); print(f'MSE: {mse:.4f}'); print(f'R-squared: {r2:.4f}')")
                        else:
                            avg_method = 'weighted'
                            ml_code_lines.append("accuracy = accuracy_score(y_test, y_pred)"); ml_code_lines.append(f"precision = precision_score(y_test, y_pred, average='{avg_method}', zero_division=0)")
                            ml_code_lines.append(f"recall = recall_score(y_test, y_pred, average='{avg_method}', zero_division=0)"); ml_code_lines.append(f"f1 = f1_score(y_test, y_pred, average='{avg_method}', zero_division=0)")
                            ml_code_lines.append("print(f'Accuracy: {accuracy:.4f}'); print(f'Precision ({avg_method}): {precision:.4f}'); print(f'Recall ({avg_method}): {recall:.4f}'); print(f'F1-Score ({avg_method}): {f1:.4f}')")
                            ml_code_lines.append("cm = confusion_matrix(y_test, y_pred)"); ml_code_lines.append("print('\\nConfusion Matrix:\\n', cm)")

                        st.session_state.ml_code = "\n".join(ml_code_lines) # Store generated code

                        # --- Execute ML Pipeline ---
                        try:
                            # 1. Prepare Data (Copy to avoid modifying original state df)
                            df_ml_exec = df_ml.copy()
                            target_col = st.session_state.ml_target
                            feature_cols = st.session_state.ml_features
                            df_ml_exec = df_ml_exec[feature_cols + [target_col]] # Select only needed cols
                            initial_rows = len(df_ml_exec)
                            df_ml_exec.dropna(subset=[target_col], inplace=True)
                            dropped_rows = initial_rows - len(df_ml_exec)
                            if dropped_rows > 0: st.warning(f"Dropped {dropped_rows} rows with missing target ('{target_col}').")

                            if df_ml_exec.empty: raise ValueError("No data after dropping missing target.")

                            X = df_ml_exec[feature_cols]
                            y = df_ml_exec[target_col]

                            if st.session_state.ml_problem_type == "Regression":
                                y = pd.to_numeric(y, errors='coerce')
                                X = X[y.notna()] ; y = y[y.notna()] # Align X and y after potential NA introduction
                                if y.empty: raise ValueError("Target column became all NA after numeric conversion.")
                            elif st.session_state.ml_problem_type == "Classification" and y.nunique() < 2:
                                raise ValueError(f"Target column '{target_col}' has less than 2 unique classes after NA drop. Cannot perform classification.")


                            # 2. Split Data
                            stratify_param = y if st.session_state.ml_problem_type == 'Classification' and y.nunique() > 1 else None
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify_param)

                            # 3. Get Preprocessor
                            pipeline_preprocessor = get_ml_pipeline(numeric_features_selected, categorical_features_selected, num_imputer_strategy, cat_imputer_strategy, cat_fill_value, scaler_option, encoder_option)

                            # 4. Get Model
                            model = get_model(model_name)
                            if model is None: raise ValueError("Could not retrieve selected model.")

                            # 5. Create Full Pipeline
                            pipeline = Pipeline(steps=[('preprocessor', pipeline_preprocessor), ('model', model)])

                            # 6. Train Pipeline
                            with st.spinner("Training model..."):
                                 pipeline.fit(X_train, y_train)
                            st.session_state.ml_pipeline = pipeline

                            # 7. Predict on Test Set
                            y_pred = pipeline.predict(X_test)

                            # 8. Evaluate
                            evaluation_results = {}
                            if st.session_state.ml_problem_type == "Regression":
                                evaluation_results['mae'] = mean_absolute_error(y_test, y_pred)
                                evaluation_results['mse'] = mean_squared_error(y_test, y_pred)
                                evaluation_results['r2'] = r2_score(y_test, y_pred)
                            else: # Classification
                                evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
                                avg_method = 'weighted'
                                evaluation_results['precision'] = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                                evaluation_results['recall'] = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                                evaluation_results['f1'] = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                                try:
                                    cm_labels = pipeline.classes_ if hasattr(pipeline, 'classes_') else sorted(y_test.unique()) # Get labels
                                    cm = confusion_matrix(y_test, y_pred, labels=cm_labels)
                                    fig_cm, ax_cm = plt.subplots()
                                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
                                    disp.plot(ax=ax_cm, cmap='Blues')
                                    plt.tight_layout(); evaluation_results['confusion_matrix_fig'] = fig_cm
                                except Exception as cm_err: st.warning(f"Could not generate confusion matrix: {cm_err}") ; evaluation_results['confusion_matrix_fig'] = None

                            st.session_state.ml_evaluation = evaluation_results
                            st.success(f"Model '{model_name}' trained and evaluated!")
                            st.rerun()

                        except Exception as ml_err:
                             st.error(f"ML process failed: {ml_err}") ; st.error(traceback.format_exc())
                             st.session_state.ml_pipeline = None ; st.session_state.ml_evaluation = {}


                    # --- 5. Display ML Results ---
                    st.subheader("5. Evaluation Results")
                    eval_results = st.session_state.get('ml_evaluation', {})
                    if not eval_results: st.caption("Train a model first.")
                    else:
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.markdown("**Metrics**")
                            if st.session_state.ml_problem_type == "Regression":
                                st.metric("R²", f"{eval_results.get('r2', 0):.4f}")
                                st.metric("MAE", f"{eval_results.get('mae', 0):.4f}")
                                st.metric("MSE", f"{eval_results.get('mse', 0):.4f}")
                            else:
                                st.metric("Accuracy", f"{eval_results.get('accuracy', 0):.4f}")
                                st.metric("Precision (W)", f"{eval_results.get('precision', 0):.4f}")
                                st.metric("Recall (W)", f"{eval_results.get('recall', 0):.4f}")
                                st.metric("F1-Score (W)", f"{eval_results.get('f1', 0):.4f}")
                        with res_col2:
                            if st.session_state.ml_problem_type == "Classification":
                                st.markdown("**Confusion Matrix**")
                                cm_fig = eval_results.get('confusion_matrix_fig')
                                if cm_fig:
                                    st.pyplot(cm_fig)
                                    cm_plot_bytes = save_plot_to_bytes(cm_fig)
                                    if cm_plot_bytes: st.download_button("Download CM Plot", cm_plot_bytes, "confusion_matrix.png", "image/png", key="dl_cm_plot")
                                else: st.caption("Not generated.")
                            elif st.session_state.ml_problem_type == "Regression":
                                 st.markdown("**Actual vs. Predicted (Sample)**")
                                 try: # Plot only if y_test and y_pred were successfully generated
                                     if 'y_test' in locals() and 'y_pred' in locals():
                                         sample_size = min(len(y_test), 100)
                                         fig_pred, ax_pred = plt.subplots()
                                         ax_pred.scatter(y_test[:sample_size], y_pred[:sample_size], alpha=0.6)
                                         min_val = min(y_test.min(), y_pred.min())
                                         max_val = max(y_test.max(), y_pred.max())
                                         ax_pred.plot([min_val, max_val], [min_val, max_val], '--r', lw=2)
                                         ax_pred.set_xlabel("Actual Values"); ax_pred.set_ylabel("Predicted Values")
                                         ax_pred.set_title(f"Actual vs. Predicted (Sample n={sample_size})")
                                         st.pyplot(fig_pred)
                                         # Plot saving needs care here as y_test/y_pred aren't in state
                                         # Could save the figure temporarily if needed for download
                                         plt.close(fig_pred)
                                     else: st.caption("Run training first.")
                                 except Exception as plot_err: st.warning(f"Plot error: {plot_err}")

                        if st.session_state.ml_code:
                             with st.expander("Show Scikit-learn Code"): st.code(st.session_state.ml_code, language='python')
                else: # Features/Target not selected
                     if not st.session_state.ml_target: st.caption("Select target column first.")
                     elif not st.session_state.ml_features: st.caption("Select feature columns first.")
            else: # Problem type not selected
                 st.info("Select a problem type to begin.")

# --- Handling Initial State (No DataFrame Loaded) ---
elif st.session_state.data_source is None:
    st.info("👈 Please choose a data source from the sidebar.")
else:
    if st.session_state.data_source == 'Upload File (CSV/Excel)': st.info("👈 Upload a file.")
    elif st.session_state.data_source == 'Connect to Database': st.info("👈 Complete DB connection steps.")
