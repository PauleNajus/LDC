from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls.i18n import i18n_patterns
from django.views.i18n import set_language
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import redirect

def is_admin_user(user):
    return user.is_authenticated and user.username == 'paubun' and user.is_staff

admin.site.login = user_passes_test(is_admin_user, login_url='core:login')(admin.site.login)

# Start with an empty urlpatterns
urlpatterns = []

# Add localized URL patterns first
urlpatterns += i18n_patterns(
    path('admin/', admin.site.urls),
    path('', include('core.urls', namespace='core')),
    prefix_default_language=True,
)

# Add non-localized URLs after the localized ones
urlpatterns += [
    path("__reload__/", include("django_browser_reload.urls")),
    path('i18n/setlang/', set_language, name='set_language'),
]

if settings.DEBUG:
    urlpatterns += [
        path("__debug__/", include("debug_toolbar.urls")),
    ]
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# Redirect root URL to default language
urlpatterns += [
    path('', lambda request: redirect(f'/{settings.LANGUAGE_CODE}/'))
] 