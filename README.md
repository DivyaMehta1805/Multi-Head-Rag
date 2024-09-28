

![Screenshot from 2024-09-28 18-47-31](https://github.com/user-attachments/assets/b53d2137-339c-4044-a461-fb1913cb1b54)

# Steps to Run the Application

### Frontend:
1. Navigate to the frontend directory.
2. Install the required dependencies:
    ```bash
    npm i
    ```
3. Start the frontend server:
    ```bash
    npm start
    ```

### Backend:
1. Ensure all dependencies are installed. These can be found in the `requirements.txt` file. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
2. To upload your document and replace the embeddings of the existing sample research paper:
    - Run the following command:
      ```bash
      python documentapi.py
      ```
    - Once executed, enter the path to your file when prompted.

3. After uploading the document, start the query API by running:
    ```bash
    python query_api.py
    ```

4. You can now ask questions about your uploaded PDF on the frontend, and the backend will provide answers based on the document.
