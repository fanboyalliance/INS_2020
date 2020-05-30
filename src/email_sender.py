import os
import smtplib

def sent_email_notification(to, message):
    login = 'alex.raswqa88@outlook.com'
    password = os.environ['Password']

    server = smtplib.SMTP('smtp.office365.com:587')
    server.ehlo()
    server.starttls()
    server.login(login, password)

    message = 'Subject: {}\n\n{}'.format('New data', message)
    server.sendmail(login, to, message)
    server.close()
