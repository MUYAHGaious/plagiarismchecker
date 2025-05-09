import sys
import sqlite3

# Get document ID from command line
if len(sys.argv) < 2:
    print("Please provide a document ID")
    print("Usage: python reset_document.py <document_id>")
    sys.exit(1)

document_id = sys.argv[1]

# Connect to the database
conn = sqlite3.connect('plagiarism_checker.db')
cursor = conn.cursor()

# Reset document status
cursor.execute(
    "UPDATE documents SET status = 'uploaded' WHERE id = ?",
    (document_id,)
)

# Check if any rows were updated
if cursor.rowcount > 0:
    print(f"Document {document_id} status reset to 'uploaded'")
else:
    print(f"Document {document_id} not found")

# Commit changes and close connection
conn.commit()
conn.close()