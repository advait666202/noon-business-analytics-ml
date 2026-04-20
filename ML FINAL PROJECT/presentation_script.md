# 🛒 Noon Daily Grocery — ML Analytics Dashboard
## Executive Presentation Script

**Pacing Note:** Each speaking part is meticulously calibrated between 190-205 words. At a comfortable, professional speaking pace of ~135 words per minute, each segment will take **exactly 1.5 minutes**.

---

### Speaker 1: Introduction & Executive Overview (193 words)
**[Action: Open the dashboard to the 'Executive Overview' tab]**

"Good morning everyone. We are thrilled to present our Machine Learning Analytics Dashboard, custom-built for the Chief Strategy Officer of Noon Daily Grocery. Today, Noon Daily faces four critical operational challenges: inefficient demand forecasting costing 18 million dirhams in food waste, rising customer churn of 30%, ineffective blanket marketing, and a very low cross-selling rate of 2.1%. To solve this, we’ve developed an industry-grade, interactive dashboard that transforms raw transactional data from 8 cities across the UAE and Saudi Arabia into actionable intelligence. 

Let me direct your attention to the **Executive Overview** page. Here, you can see our high-level KPIs and interactive revenue trends at a glance. You'll immediately notice the significant 18 to 25 percent revenue surge during the Ramadan period. Using our interactive filters, executives can drill down by city, category, and date. This dashboard forms the foundation for data-driven decision-making, allowing leaders to quickly assess business health before diving into predictive insights. We’ve broken our ML solutions into four core modules—Regression, Classification, Clustering, and Association. To walk you through our first predictive model, demand forecasting, I will now hand it over to my colleague."

---

### Speaker 2: Demand Forecasting / Regression (205 words)
**[Action: Click on the 'Regression' tab]**

"Thank you. Let’s navigate to the **Regression** tab, where we tackle our first major challenge: Poor demand forecasting. As mentioned, Noon currently suffers around 18 million dirhams a year in food waste and stockouts due to a current 20 to 25 percent forecast error margin.

To address this, we developed a robust weekly demand forecasting engine. We evaluated three regression models: Linear, Ridge, and Lasso. Our dashboard dynamically displays their performance metrics, including R-squared, RMSE, and MAPE, alongside Actual-versus-Predicted visualization plots. By implementing these models, we successfully reduced the Mean Absolute Percentage Error to under 15 percent. From a business perspective, this precision translates directly to potential savings of 4 to 6 million dirhams annually, simply by better aligning our perishable inventory with actual demand. 

Furthermore, our dashboard includes an interactive revenue scenario simulator. This allows the strategy team to adjust pricing or promotional parameters and instantly visualize the predicted impact on weekly demand. This module transitions our supply chain from a reactive guessing game into a proactive operation. By ensuring we have the right products available exactly when the customer needs them, we naturally improve retention. I’ll now pass it over to discuss Customer Churn."

---

### Speaker 3: Customer Churn Prediction / Classification (201 words)
**[Action: Click on the 'Classification' tab]**

"Thank you. Now, please look at the **Classification** tab, which focuses on predicting customer churn. Currently, we face a 30 percent churn rate, and mathematically, acquiring a new customer costs 8 times more than retaining an existing one. 

We deployed three robust classifiers: Logistic Regression, Random Forest, and Gradient Boosting. We evaluate these models dynamically through ROC curves and confusion matrices directly on the dashboard. Our analysis uncovered two massive business insights. First, the feature 'days since last purchase' is the absolute number one predictor of churn, with a critical risk cliff occurring exactly at the 45-day mark. Second, customers who leave no delivery ratings churn 2.3 times faster than those who do—proving that missing data is in itself a powerful signal. 

To make this actionable, we incorporated an interactive threshold slider and an individual churn predictor tool. Instead of waiting for users to leave, Noon can now automatically flag at-risk customers as they approach that 45-day threshold. We calculate that a proactive retention strategy based on these specific predictions yields over 2.1 million dirhams in net benefit per 10,000 customers. Next, we will discuss optimizing marketing spend through Customer Segmentation."

---

### Speaker 4: Customer Segmentation / Clustering (194 words)
**[Action: Click on the 'Clustering' tab]**

"Thank you. Moving to the **Clustering** tab, we address the challenge of our 22 million dirham marketing budget, which is currently being spent uniformly across our entire customer base. To optimize our Return on Ad Spend, we implemented K-Means clustering.

Through the dashboard, executives can interactively explore the clustering process. We've included visualizations for the Elbow and Silhouette methods to justify our algorithm parameters, alongside beautiful 3-D PCA scatter plots that visually separate our users. Our analysis clearly identified five distinct customer segments with unique spending behaviors and lifetime values. 

We translated these mathematical clusters into actionable business labels, visually represented through detailed radar profiles. Instead of sending the same promotions to everyone, Noon can now tailor its campaigns. For instance, high-value loyalists can receive VIP early-access perks, while price-sensitive occasional shoppers receive targeted discount codes. This segmentation ends the financial inefficiency of blanket marketing, allowing Noon to dynamically allocate budget where it will drive the highest margin. By understanding exactly 'who' is buying, we can better understand 'what' they are buying together, which brings us to Market Basket Analysis. I'll hand over for the final segment."

---

### Speaker 5: Market Basket Analysis & Conclusion (198 words)
**[Action: Click on the 'Association Rules' tab]**

"Thank you. Finally, let’s explore the **Association Rules** tab for Market Basket Analysis. Our overarching goal here is to fix our low cross-selling click-through rate, which currently sits at only 2.1 percent against an industry average of 8 to 12 percent. 

We ran the Apriori algorithm on over 8,000 grocery transactions to uncover hidden purchasing patterns. On this page, you can configure the support and confidence thresholds to instantly generate the top product rules ranked by Lift. One standout insight we've identified is the strong continuous association between Bread, Dairy, and Eggs. Because these items are frequently co-purchased, we recommend an automated pricing strategy, such as offering a 10 percent combo discount when they are bundled.

Integrating these algorithm-backed recommendations into the checkout flow will drastically increase our basket sizes. 

In conclusion, this interactive dashboard gives Noon Daily's executive team a comprehensive tool to reduce waste, lower churn, optimize marketing spend, and boost average order values. It successfully bridges the gap between raw data and measurable business impact, positioning Noon for scalable, hyper-efficient growth. Thank you very much for your time, we are now open for any questions."
