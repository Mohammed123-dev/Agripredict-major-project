@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agronomy - Crop Recommendation'

    if request.method == 'POST':
        try:
            # Get form inputs
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city = request.form.get("city")

            # Fetch weather safely
            try:
                weather = weather_fetch(city)  # Use your weather_fetch function
            except Exception as e:
                print(f"[ERROR] Weather fetch failed: {e}")
                weather = None

            if weather:
                temperature, humidity = weather
            else:
                # Default values if weather fetch fails
                temperature, humidity = 25.0, 50.0
                print(f"[WARNING] Using default weather values for {city}")

            # Prepare input for model
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Make prediction
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        except ValueError as ve:
            error_msg = f"Invalid input: {ve}"
            return render_template('try_again.html', title=title, error=error_msg)
        except Exception as e:
            error_msg = f"Something went wrong: {e}"
            return render_template('try_again.html', title=title, error=error_msg)
