from sklearn.model_selection import train_test_split
import streamlit as st
import pickle




def start_training(dataframe,target):
        

        # Assuming X contains features and y contains target values
        st.write("In train models")
        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        # Split the data into training and testing sets (e.g., 80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Display the shapes of the resulting sets
        


        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import accuracy_score

       
        RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
        RandomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=42)
        list_of_models = [RandomForestClassifier]

        st.write("Initialized models")
        # Train each model using the training set
        for model in list_of_models:
            model.fit(X_train, y_train)
            st.write(f"Trained model {model}")

    # Evaluate each model on the testing set (optional)
        for model in list_of_models:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model: {model.__class__.__name__}, Accuracy: {accuracy}")


            with open('model.pkl', 'wb') as f:
                pickle.dump(list_of_models[0], f)
            with open('model.pkl', 'rb') as f:
                data = pickle.load(f)
                data = pickle.dumps(data)              

            st.download_button('Download Model',file_name= 'model.pkl',data = data)
