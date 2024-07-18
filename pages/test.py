import streamlit as st

# Initialize session state if it doesn't already exist
if 'show_text' not in st.session_state:
    st.session_state.show_text = True

# Function to toggle the visibility of the text
def toggle_text():
    st.session_state.show_text = not st.session_state.show_text

# Display the button to toggle the text
st.button("Toggle Text", on_click=toggle_text)

# Conditionally display the text based on session state
if st.session_state.show_text:
    st.write("This is the text to be toggled.")
import streamlit as st

# Create two columns
col1, col2 = st.columns(2)

# Initialize result variable
result = None

# Perform operation and condition check in Column 1
with col1:
    st.write("Column 1")
    num1 = st.number_input("Enter the first number", key="num1", value=2)
    num2 = st.number_input("Enter the second number", key="num2", value=5)
    
    if st.button("Calculate", key="calculate"):
        result = num1 * num2  # Example operation
        condition = result > 10  # Example condition
        
        if condition:
            st.write("The result is greater than 10.")
        else:
            st.write("The result is 10 or less.")

# Display result in Column 2 based on the result from Column 1
with col2:
    st.write("Column 2")
    if result is not None:
        st.write(f"The result of the multiplication is: {result}")
    else:
        st.write("No calculation performed yet.")
