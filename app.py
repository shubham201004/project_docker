from streamlit import set_page_config, navigation, columns, markdown, image, container
import os
from st_pages import add_page_title, get_nav_from_toml

current_dir = os.path.dirname(os.path.abspath(__file__))
# assets_path = os.path.join(current_dir,'Assets')

set_page_config(
    layout="wide",
    page_title="CDAC's AI-Powered Plant Disease Detection and Suggestion System ",
    page_icon="🌱" 
)

Navigation = get_nav_from_toml(
    "page_sections.toml"
)

pages = navigation(Navigation)

add_page_title(pages)

pages.run()
