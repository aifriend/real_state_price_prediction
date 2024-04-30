import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AnalyticService:
    SHOW = True

    @staticmethod
    def show_rev_per_day(df):
        # Convert the 'date' column to datetime if it's not already
        df['date'] = pd.to_datetime(df['date'])

        # Create a new DataFrame with the count of reviews for each day
        daily_reviews = df.groupby(df['date'].dt.date).size().reset_index(name='count')

        # Create a figure and axis using Seaborn
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create a line plot using Seaborn
        sns.lineplot(data=daily_reviews, x='date', y='count', ax=ax)

        # Set the title and labels for the plot
        ax.set_title('Daily Number of Reviews')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Reviews')

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=45)

        # Show the plot
        if AnalyticService.SHOW:
            plt.show()

    @staticmethod
    def show_rev_length(df):
        # Calculate the length of each review
        df['review_length'] = df['comments'].apply(len)

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the distribution of review lengths using Seaborn's histplot
        sns.histplot(data=df, x='review_length', ax=ax, kde=True)

        # Set the title and labels
        ax.set_title('Distribution of Review Lengths')
        ax.set_xlabel('Review Length')
        ax.set_ylabel('Count')

        # Display the plot
        if AnalyticService.SHOW:
            plt.show()
