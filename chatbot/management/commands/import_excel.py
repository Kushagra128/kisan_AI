import os
import pandas as pd
from django.core.management.base import BaseCommand
from chatbot.models import AgriculturalAdvice

class Command(BaseCommand):
    help = 'Imports agricultural data from adv_data.xlsx into the PostgreSQL database'

    def handle(self, *args, **kwargs):
        file_path = os.path.join('chatbot', 'adv_data.xlsx')
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f"File {file_path} does not exist."))
            return

        self.stdout.write("Loading dataset from Excel...")
        df_raw = pd.read_excel(file_path)
        df = (
            df_raw[["problem", "solution", "cropname"]]
            .dropna(subset=["cropname", "problem"])
            .fillna("")
            .copy()
        )
        df["cropname"] = df["cropname"].astype(str).str.strip()
        df["problem"]  = df["problem"].astype(str).str.strip()
        df["solution"] = df["solution"].astype(str).str.strip()
        df = df[df["problem"].str.len() > 5].reset_index(drop=True)

        self.stdout.write("Deleting existing records...")
        AgriculturalAdvice.objects.all().delete()

        self.stdout.write("Importing records...")
        records = [
            AgriculturalAdvice(
                cropname=row["cropname"],
                problem=row["problem"],
                solution=row["solution"]
            )
            for _, row in df.iterrows()
            if row["problem"] and row["solution"]
        ]
        
        AgriculturalAdvice.objects.bulk_create(records)
        self.stdout.write(self.style.SUCCESS(f"Successfully imported {len(records)} records."))
