from django.test import TestCase
from selenium import webdriver
import pytest
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time


driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

driver.get("http://127.0.0.1:8000/")

time.sleep(2)
driver.close()
driver.quit()

# Create your tests here.
   