from django.contrib.auth import authenticate, login 
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import *
from .forms import *
from django.shortcuts import get_object_or_404
from django.contrib.postgres.search import TrigramSimilarity
import re
from django.conf import settings
from django.core.mail import send_mail
from django.http import HttpResponse
from django.db.models import Q
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse
from django.contrib.admin.views.decorators import staff_member_required

def welcome(request):
    return render(request, template_name="welcome.html")

def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken. Please choose another.')
        else:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            login(request, user)  # Log the user in after signup
            messages.success(request, 'Account created successfully!')
            return redirect('login')  # Redirect to a login

    return render(request, 'signup.html')

def user_login(request):  # Renamed to avoid conflict
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Use Django's login function
            return redirect('welcome')  # Redirect to the welcome page after login
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    return redirect('welcome') 

@login_required
def profile(request):
    # Get the current user's profile or create one if it doesn't exist
    user_profile, created = Profile.objects.get_or_create(user=request.user)

    context = {
        'user': request.user,
        'profile': user_profile,
    }
    return render(request, 'profile.html', context)

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Profile
from .forms import ProfileForm  # Use your existing profile form

@login_required
def edit_profile(request):
    # Get the current user's profile
    profile = get_object_or_404(Profile, user=request.user)

    if request.method == 'POST':
        # Bind the form to the user's existing profile instance
        form = ProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            # Save the form without committing to override account_type
            updated_profile = form.save(commit=False)

            # Automatically set account_type to 'admin' if the user is staff
            if request.user.is_staff:
                updated_profile.account_type = 'admin'
            
            # Save the updated profile to the database
            updated_profile.save()
            return redirect('profile')  # Redirect to the profile page after editing
    else:
        # Prefill the form with the existing profile data
        form = ProfileForm(instance=profile)

    return render(request, 'edit_profile.html', {'form': form})


def complaints(request):
    complaints = complaint.objects.all()
    context = {
        'complaints' : complaints,
    }
    return render(request, template_name="complaints.html", context=context)

def contact_us(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Get cleaned data from the form
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            phone_number = form.cleaned_data['phone_number']
            message = form.cleaned_data['message']
            subject = form.cleaned_data['subject']

            # Store the message in the database
            ContactMessage.objects.create(
                name=name,
                email=email,
                phone_number=phone_number,
                message=message,
                subject=subject
            )

            # After successful form submission, redirect to a thank-you page
            return render(request, 'thank_you.html')

    else:
        form = ContactForm()

    return render(request, 'contact_us.html', {'form': form})


def confirm_delete_complaint(request, restaurant_id, complaint_id):
    complaint_instance = get_object_or_404(complaint, id=complaint_id)
    
    # Pass the complaint details to the confirmation page
    context = {'complaint': complaint_instance}
    return render(request, 'confirm_delete_complaints.html', context)

def delete_complaint(request, restaurant_id, complaint_id):
    complaint_instance = get_object_or_404(complaint, id=complaint_id)

    if request.method == 'POST':
        # Perform the deletion
        complaint_instance.delete()
        return redirect('complaints', restaurant_id=restaurant_id)
    else:
        # Redirect if not POST
        return HttpResponse(status=405)  # Method Not Allowed

def create_report(request, id):
    restaurant_detail = get_object_or_404(restaurant, id=id)
    
    if request.method == 'POST':
        description = request.POST.get('complaint_description')
        issue = request.POST.get('issue')  # Get the complaint issue from the dropdown
        complaint_pics = request.FILES.getlist('complaint_pic')  # Retrieve all uploaded files
        
        if description:
            # Create the complaint instance
            new_complaint = complaint.objects.create(
                restaurant_Name=restaurant_detail,
                user_Name=request.user if request.user.is_authenticated else None,
                complaint_Description=description,
                complaint_topic=issue,
                issued_date=timezone.now(),
                update_date=timezone.now()
            )

            # Assign images if they are provided, up to 3
            if len(complaint_pics) > 0:
                new_complaint.image1 = complaint_pics[0]
            if len(complaint_pics) > 1:
                new_complaint.image2 = complaint_pics[1]
            if len(complaint_pics) > 2:
                new_complaint.image3 = complaint_pics[2]

            # Save the complaint with images
            new_complaint.save()

            #messages.success(request, 'Your complaint has been successfully submitted!')
            return redirect('res_details', id=id)
        else:
            messages.error(request, 'Please provide a description for your complaint.')

    context = {
        'restaurant': restaurant_detail
    }
    return render(request, 'report_form.html', context)


def restaurants(request):
    restaurants = restaurant.objects.all()
    context = {
        'restaurants' : restaurants,
    }
    return render(request, template_name="restaurants.html", context=context)

def update_restaurant(request, restaurant_id):
    # Get the specific restaurant
    restaurants = get_object_or_404(restaurant, id=restaurant_id)

    if request.method == 'POST':
        # Bind data to the form
        form = RestaurantForm(request.POST, request.FILES, instance=restaurants)
        if form.is_valid():
            form.save()
            messages.success(request, "Restaurant details updated successfully.")
            return redirect('restaurants')  # Redirect to the restaurant list
    else:
        # Prepopulate the form with the restaurant's current details
        form = RestaurantForm(instance=restaurants)

    return render(request, 'update_restaurant.html', {'form': form, 'restaurant': restaurants})


def delete_restaurant(request, restaurant_id):
    restaurant_instance = get_object_or_404(restaurant, id=restaurant_id)
    restaurant_instance.delete()
  
    return redirect('restaurants')


def res_details(request, id):
    restaurant_detail = get_object_or_404(restaurant, id=id)
    context = {
        'restaurant' : restaurant_detail
    }
    return render(request, template_name="res_details.html", context=context)



def restaurants_search(request):
    form = RestaurantSearchForm(request.GET)  # Initialize with GET data
    results = []
    
    if form.is_valid():
        query = form.cleaned_data['query']
        
        # Use regex for flexible partial matching
        results = restaurant.objects.filter(Q(name__iregex=r'{}'.format(re.escape(query))))
        
        return render(request, 'restaurants.html', {'restaurants': results, 'query': query})
    
    # Render the initial search form in case of GET request without search
    return render(request, 'search_res.html', {'form': form})

def restaurant_complaints(request, restaurant_id):
    complaints = complaint.objects.filter(restaurant_Name_id=restaurant_id)
    return render(request, 'complaints.html', {'complaints': complaints})


def complaint_details(request, complaint_id):
    complaint_detail = get_object_or_404(complaint, id=complaint_id)
    context = {
        'complaint': complaint_detail
    }
    return render(request,template_name='complaint_detail.html', context=context)


def update_complaint_status(request, complaint_id):
    complaint_instance = get_object_or_404(complaint, id=complaint_id)

    if request.method == 'POST':
        status = request.POST.get('status')
        complaint_instance.status = status
        complaint_instance.save()

        messages.success(request, 'Complaint status updated successfully!')
        

    context = {
        'complaint': complaint_instance,
    }
    return render(request, 'update_complaint_status.html', context)

def delete_complaint(request, restaurant_id, complaint_id):
    complaint_instance = get_object_or_404(complaint, id=complaint_id)
    complaint_instance.delete()
    #messages.success(request, "Complaint deleted successfully!")
    return redirect('restaurant_complaints', restaurant_id=restaurant_id)



# View to add pending restaurant
def add_pending_restaurant(request):
    if request.method == 'POST':
        form = PendingRestaurantForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            
            return redirect('restaurants')  # Redirect to main restaurant page
    else:
        form = PendingRestaurantForm()
    return render(request, 'add_pending_restaurant.html', {'form': form})


# Admin-only view to approve pending restaurants
def approve_restaurant(request, pending_id):
    pending_restaurant = get_object_or_404(PendingRestaurant, id=pending_id)
    approved_restaurant = restaurant.objects.create(
        name=pending_restaurant.name,
        trade_License_Number=pending_restaurant.trade_License_Number,
        location=pending_restaurant.location,
        address=pending_restaurant.address,
        cuisine=pending_restaurant.cuisine,
        no_of_Employee=pending_restaurant.no_of_Employee,
        image=pending_restaurant.image
    )
    pending_restaurant.delete()
    
    return redirect('pending_restaurants_list')  # Custom page for pending list

def reject_restaurant(request, pending_id):
    pending_restaurant = get_object_or_404(PendingRestaurant, id=pending_id)
    pending_restaurant.delete()  # Just delete without assigning to a variable
    return redirect('pending_restaurants_list')  # Redirect to the pending restaurants list page


def pending_restaurants_list(request):
    pending_restaurants = PendingRestaurant.objects.filter(approved=False)
    return render(request, 'pending_restaurants_list.html', {'pending_restaurants': pending_restaurants})

