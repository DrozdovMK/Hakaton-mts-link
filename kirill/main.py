import gspread
from oauth2client.service_account import ServiceAccountCredentials


scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('key.json', scope)
client = gspread.authorize(creds)
spreadsheet = client.open('MTC-link')
worksheet = spreadsheet.get_worksheet(0)
data = worksheet.get_all_values()
response = []
name = []
organisation = []
data.pop(0)
for elem in data:
    name.append(elem[2])
    organisation.append(elem[3])
    response.append(elem[1].lower())
print(response)
print(name)
print(organisation)
print(len(response), len(name), len(organisation))