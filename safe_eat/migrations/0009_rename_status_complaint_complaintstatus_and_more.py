# Generated by Django 5.1.2 on 2024-10-14 20:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('safe_eat', '0008_rename_trade_license_no_restaurant_trade_license_number_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='complaint',
            old_name='status',
            new_name='complaintStatus',
        ),
        migrations.RenameField(
            model_name='complaint',
            old_name='restaurant',
            new_name='restaurantName',
        ),
        migrations.RenameField(
            model_name='complaint',
            old_name='user',
            new_name='userName',
        ),
    ]
