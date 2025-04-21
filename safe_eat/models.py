from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Profile(models.Model):

    ACCOUNT_TYPES = [
        ('customer', 'customer'),
        ('admin', 'admin'),
    ]
     
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    bio = models.TextField(max_length=500, blank=True)
    address = models.CharField(max_length=100, blank=True)
    account_type = models.CharField(max_length=10, choices=ACCOUNT_TYPES, default= 'customer')
    profile_picture = models.ImageField(upload_to='static/images', blank=True, null=True, default= 'static/images/user-profile.png')
    email = models.EmailField(max_length=254, null=True, blank=True)

    def save(self, *args, **kwargs):
        if self.user.is_staff:
            self.account_type = 'admin'
        super().save(*args, **kwargs)

    def __str__(self):
        return self.user.username
    

class restaurant(models.Model):
    name = models.CharField(max_length=200)
    trade_License_Number = models.CharField(max_length=500, null=True)
    location = models.CharField(max_length=100)
    address = models.CharField(max_length=500, blank=True)
    cuisine = models.CharField(max_length=500, blank=True, null=True)
    no_of_Employee = models.IntegerField(blank=True, null=True)
    image = models.ImageField(upload_to='static/images',blank=True, null=True, default='static/images/no_logo.jpg')

    def __str__(self):
        return self.name
    
class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    phone_number = models.CharField(max_length=15, blank=True)
    message = models.TextField()
    subject = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.name} - {self.subject}"
    
class complaint(models.Model):
    user_Name = models.ForeignKey(User, on_delete=models.CASCADE, blank=True, null=True)
    restaurant_Name = models.ForeignKey(restaurant, on_delete=models.CASCADE, blank=True, null=True)
    issued_date = models.DateField(auto_now_add=True, auto_now=False)
    update_date = models.DateField(auto_now_add=True, auto_now=False)
    complaint_Description = models.TextField(max_length=1000, blank=False, null=True)
    complaint_topic = models.CharField(max_length=255, null=True, blank=True)

    image1 = models.ImageField(upload_to='static/images', blank=True, null=True)
    image2 = models.ImageField(upload_to='static/images', blank=True, null=True)
    image3 = models.ImageField(upload_to='static/images', blank=True, null=True)

    STATUS_CHOICES = [
        ('resolved', 'Resolved'),
        ('in_process', 'In Process'),
        ('fake', 'Fake'),
        ('pending', 'Pending'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return self.restaurant_Name.name
    

class PendingRestaurant(models.Model):
    name = models.CharField(max_length=200)
    trade_License_Number = models.CharField(max_length=500, null=True, blank=True)
    location = models.CharField(max_length=100)
    address = models.CharField(max_length=500, blank=True)
    cuisine = models.CharField(max_length=500, blank=True, null=True)
    no_of_Employee = models.IntegerField(blank=True, null=True)
    image = models.ImageField(upload_to='static/images', blank=True, null=True, default='static/images/no_logo.jpg')
    submitted_date = models.DateTimeField(auto_now_add=True)
    approved = models.BooleanField(default=False)  # Tracks if the restaurant is approved

    def __str__(self):
        return self.name
    
