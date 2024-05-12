import smtplib

# Create an SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)

# Start TLS for security
s.starttls()

# Authentication 
s.login("asaninnovators48@gmail.com", "Block@#78")

# Message to be sent
message = "Message_you_need_to_send"

# Sending the mail
s.sendmail("thiruveedhulabalaji3@gmail.com", "tbalaji8822@gmail.com", message)

# Terminate the session
s.quit()
