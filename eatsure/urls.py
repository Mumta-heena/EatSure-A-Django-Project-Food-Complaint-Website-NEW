from django.contrib import admin
from django.urls import path
from safe_eat import views as v
from django.conf.urls.static import static
from django.conf import settings
from safe_eat.views import *



urlpatterns = [
    path('admin/', admin.site.urls),

    path('', v.welcome, name='welcome'),

    path('profile/', v.profile, name='profile'),

    path('profile/edit/', v.edit_profile, name='edit_profile'),

    path('complaints/', v.complaints, name='complaints'),
   
    path('restaurants/<int:id>/', v.res_details, name='res_details'),

    path('restaurant/update/<int:restaurant_id>/', v.update_restaurant, name='update_restaurant'),

    path('restaurants/', v.restaurants, name='restaurants'),
    
    path('contacts/', v.contact_us, name='contact_us'),

    path('search/', v.restaurants_search, name='search'),

    path('login/', v.user_login, name='login'),

    path('logout/', v.logout_view, name='logout'),

    path('signup/', v.signup, name='signup'),

    path('restaurants/<int:id>/report/', v.create_report, name='create_report'),

    path('restaurant/<int:restaurant_id>/delete/', v.delete_restaurant, name='delete_restaurant'),

    path('restaurant/<int:restaurant_id>/complaints/', v.restaurant_complaints, name='restaurant_complaints'),

    path('complaint/<int:complaint_id>/complaint_details/', v.complaint_details, name='complaint_detail'),
    
    path('complaint/<int:complaint_id>/update_status/', v.update_complaint_status, name='update_complaint_status'),
    
    path('restaurant/<int:restaurant_id>/complaint/<int:complaint_id>/confirm_delete/', v.confirm_delete_complaint, name='confirm_delete_complaint'),

    path('restaurant/<int:restaurant_id>/complaint/<int:complaint_id>/delete/', v.delete_complaint, name='delete_complaint'),

    path('add-restaurant/', v.add_pending_restaurant, name='add_pending_restaurant'),

    path('pending_restaurants_list/', v.pending_restaurants_list, name='pending_restaurants_list'),

    path('approve-restaurant/<int:pending_id>/', v.approve_restaurant, name='approve_restaurant'),

    path('reject-restaurant/<int:pending_id>/', v.reject_restaurant, name='reject_restaurant'),

    
]




urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

