from django.urls import path
from . import views

urlpatterns = [
    # Halaman Utama & Simulasi Menu Dashboard
    path('', views.home_page, name='home_page'),
    path('home', views.home_page, name='home_page'),
    path('daya_beli', views.daya_beli_page, name='daya_beli_page'),
    path('forecasting', views.forecasting_page, name='forecasting_page'),
    
    # API Backend Pengeksekusi Model Ridge
    path('api/simulate-daya-beli', views.simulate_daya_beli, name='api_simulate_daya_beli'),
]