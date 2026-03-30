@app.route('/predict_analysis', methods=['GET', 'POST'])
def predict_analysis():

    if "verified_phone" not in session:
        return redirect("/")

    phone = session.get("verified_phone")

    # =========================
    # SHOW PAGE (GET)
    # =========================
    if request.method == "GET":
        return render_template("complaint.html")

    try:

        post_id = request.form.get("post_id")

        if not post_id:
            return render_template("complaint.html", error="Enter Post ID")

        db = get_db_connection()
        cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # =========================
        # FETCH POST DETAILS
        # =========================
        cursor.execute("SELECT * FROM post WHERE post_id=%s", (post_id,))
        post_data = cursor.fetchone()

        if not post_data:
            cursor.close()
            db.close()
            return render_template("complaint.html", error="Invalid Post ID")

        area = post_data["area"]
        employee_name = post_data["employee_name"]

        # =========================
        # IMAGE HANDLING
        # =========================
        import base64
        from io import BytesIO

        captured_image = request.form.get("captured_image")
        image_file = request.files.get("image")

        if captured_image:
            image_data = captured_image.split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            filename = f"{post_id}_camera.png"

        elif image_file and image_file.filename != "":
            image = Image.open(image_file).convert("RGB")
            filename = f"{post_id}_{image_file.filename}"

        else:
            cursor.close()
            db.close()
            return render_template("complaint.html", error="Capture or Upload Image")

        # Resize image (reduces memory)
        image = image.resize((224, 224))

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(filepath)

        # =========================
        # CNN PREDICTION
        # =========================
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        fault = class_names[predicted.item()]
        confidence_score = round(confidence.item() * 100, 2)

        # Reject low confidence
        if confidence_score < 0:
            cursor.close()
            db.close()
            return render_template(
                "complaint.html",
                error="Invalid Image! Please upload a clear streetlight image."
            )       

        # =========================
        # USER INPUT
        # =========================
        fault1 = request.form.get("fault1")
        fault2 = request.form.get("fault2")
        fault3 = request.form.get("fault3")
        suggestion = request.form.get("suggestion")

        # =========================
        # INSERT INTO DATABASE
        # =========================
        cursor.execute("""
            INSERT INTO complaints
            (phone, post_id, area, employee_name, cnn_result, confidence,
             fault1, fault2, fault3, suggestion, image_path, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            phone,
            post_id,
            area,
            employee_name,
            fault,
            confidence_score,
            fault1,
            fault2,
            fault3,
            suggestion,
            "uploads/" + filename,
            "Pending"
        ))

        complaint_id = cursor.fetchone()[0]

        db.commit()
        cursor.close()
        db.close()

        # =========================
        # RETURN RESULT PAGE
        # =========================
        return render_template(
            "complaint.html",
            success="Complaint Submitted Successfully!",
            fault=fault,
            confidence=confidence_score,
            area=area,
            employee_name=employee_name,
            image_path=url_for("static", filename="uploads/" + filename),
            id=complaint_id
        )

    except Exception as e:

        print("Prediction Error:", e)

        return render_template(
            "complaint.html",
            error="Server Error during prediction. Please try again."
        )