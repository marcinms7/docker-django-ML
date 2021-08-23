
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "neural_network.settings")

import django
django.setup()

from django.core.management import call_command

from django.contrib.auth.models import User

User.objects.create_superuser('admin', 'admin@example.com', 'pass')