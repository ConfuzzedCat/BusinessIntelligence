import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_option_menu import option_menu


import pandas as pd
import numpy as np

from PIL import Image



st.set_page_config(
    page_title="Streamlit MiniProject4",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={        
    }
    
)