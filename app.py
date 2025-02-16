from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this to a strong secret key

# Initialize SQLite3 Database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    # User table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE NOT NULL, 
                  email TEXT UNIQUE NOT NULL, 
                  password TEXT NOT NULL)''')
    
    # History table
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  query TEXT NOT NULL,
                  papers TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

init_db()  # Call it once to create the database
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("dashboard"))  # If logged in, go to dashboard
    return redirect(url_for("login"))

# ---------------- Signup Route ---------------- #
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not username or not email or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                      (username, email, hashed_password))
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or Email already exists. Try again!", "danger")
        finally:
            conn.close()

    return render_template("signup.html")

# ---------------- Login Route ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Try again!", "danger")

    return render_template("login.html")

# ---------------- Dashboard Route ---------------- #
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please log in first!", "warning")
        return redirect(url_for("login"))

    return render_template("dashboard.html", username=session["username"])

# ---------------- Retrieve History ---------------- #
@app.route("/get_history")
def get_history():
    if "user_id" not in session:
        return jsonify([])  # Return empty if not logged in

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT query, papers, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC", 
              (session["user_id"],))
    history = [{"query": row[0], "papers": row[1], "timestamp": row[2]} for row in c.fetchall()]
    conn.close()
    return jsonify(history)

# ---------------- Process Query & Save to DB ---------------- #
@app.route("/process_query", methods=["POST"])
def process_query():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 403

    data = request.json
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Simulate research paper retrieval
    research_papers = ["Paper1", "Paper2", "Paper3"]  # Placeholder

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (user_id, query, papers) VALUES (?, ?, ?)", 
              (session["user_id"], query, ", ".join(research_papers)))
    conn.commit()
    conn.close()

    return jsonify({"query": query, "papers": research_papers})

# ---------------- Logout Route ---------------- #
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out!", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
