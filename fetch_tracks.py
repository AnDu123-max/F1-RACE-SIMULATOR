import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

races = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 
    'Imola', 'Monaco', 'Canada', 'Spain', 'Austria', 'UK', 
    'Hungary', 'Belgium', 'Netherlands', 'Monza', 'Azerbaijan', 'Singapore', 
    'United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

# Map common country/track names to official F1 circuit naming for images
f1_url_mappings = {
    'Bahrain': 'Bahrain',
    'Saudi Arabia': 'Saudi_Arabia',
    'Australia': 'Australia',
    'Japan': 'Japan',
    'China': 'China',
    'Miami': 'Miami',
    'Imola': 'Emilia_Romagna',
    'Monaco': 'Monaco',
    'Canada': 'Canada',
    'Spain': 'Spain',
    'Austria': 'Austria',
    'UK': 'Great_Britain',
    'Hungary': 'Hungary',
    'Belgium': 'Belgium',
    'Netherlands': 'Netherlands',
    'Monza': 'Italy',
    'Azerbaijan': 'Baku',
    'Singapore': 'Singapore',
    'United States': 'USA',
    'Mexico': 'Mexico',
    'Brazil': 'Brazil',
    'Las Vegas': 'Las_Vegas',
    'Qatar': 'Qatar',
    'Abu Dhabi': 'Abu_Dhabi'
}

configs = {
    "Bahrain": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Bahrain_Circuit.png.transform/2col/image.png",
    "Saudi Arabia": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Saudi_Arabia_Circuit.png.transform/2col/image.png",
    "Australia": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Australia_Circuit.png.transform/2col/image.png",
    "Japan": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Japan_Circuit.png.transform/2col/image.png",
    "China": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/China_Circuit.png.transform/2col/image.png",
    "Miami": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Miami_Circuit.png.transform/2col/image.png",
    "Imola": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Emilia_Romagna_Circuit.png.transform/2col/image.png",
    "Monaco": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Monaco_Circuit.png.transform/2col/image.png",
    "Canada": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Canada_Circuit.png.transform/2col/image.png",
    "Spain": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Spain_Circuit.png.transform/2col/image.png",
    "Austria": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Austria_Circuit.png.transform/2col/image.png",
    "UK": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Great_Britain_Circuit.png.transform/2col/image.png",
    "Hungary": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Hungary_Circuit.png.transform/2col/image.png",
    "Belgium": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Belgium_Circuit.png.transform/2col/image.png",
    "Netherlands": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Netherlands_Circuit.png.transform/2col/image.png",
    "Monza": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Italy_Circuit.png.transform/2col/image.png",
    "Azerbaijan": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Baku_Circuit.png.transform/2col/image.png",
    "Singapore": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Singapore_Circuit.png.transform/2col/image.png",
    "United States": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/USA_Circuit.png.transform/2col/image.png",
    "Mexico": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Mexico_Circuit.png.transform/2col/image.png",
    "Brazil": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Brazil_Circuit.png.transform/2col/image.png",
    "Las Vegas": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Las_Vegas_Circuit.png.transform/2col/image.png",
    "Qatar": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Qatar_Circuit.png.transform/2col/image.png",
    "Abu Dhabi": "https://media.formula1.com/image/upload/f_auto/q_auto/v1677244985/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/Abu_Dhabi_Circuit.png.transform/2col/image.png",
}

print(json.dumps(configs, indent=4))
