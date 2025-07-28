import logging
import re
import requests
import urllib.parse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    """Data class representing a dataset recommendation"""
    title: str
    description: str
    source: str  # "sample", "uci"
    domain: str  # "tabular", "timeseries", "multimodal"
    url: str
    download_url: str
    size_mb: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    file_format: str = "csv"
    tags: List[str] = None
    difficulty: str = "beginner"  # "beginner", "intermediate", "advanced"
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DatasetRecommendationTool:
    """Tool for finding and recommending datasets from online sources"""
    
    def __init__(self):
        """Initialize the dataset recommendation tool"""
        self.sources = {
            "uci": UCIMLRepoSource(),        # UCI ML Repository scraping
            # "kaggle": KaggleDatasetSource()  # Kaggle dataset search
        }
        logger.info("🔍 Dataset recommendation tool initialized")
    
    def recommend_datasets(self, user_query: str, domain: str = "tabular", max_results: int = 5) -> Dict[str, Any]:
        """
        Main method to get dataset recommendations
        
        Args:
            user_query: User's original query/request
            domain: Detected domain (tabular, timeseries, multimodal)
            max_results: Maximum number of datasets to recommend
            
        Returns:
            Dictionary with formatted results for agent
        """
        try:
            logger.info(f"🔍 Searching for datasets: query='{user_query}', domain='{domain}'")
            print(f"🔍 DATASET SEARCH: Query='{user_query}', Domain='{domain}'")
            
            # Extract simple keywords from query (for fallback)
            keywords = self._extract_keywords(user_query)
            logger.info(f"📊 Query analysis: '{user_query}' -> Keywords: {keywords}")
            
            # Collect datasets from all sources
            all_datasets = []
            
            for source_name, source in self.sources.items():
                try:
                    logger.info(f"🔎 Searching {source_name} with semantic matching...")
                    datasets = source.search_datasets(
                        keywords=[user_query],  # Pass full query for semantic analysis
                        domain=domain,
                        max_results=max_results
                    )
                    all_datasets.extend(datasets)
                    logger.info(f"✅ Found {len(datasets)} datasets from {source_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Error searching {source_name}: {e}")
                    continue
            
            # Take top results (sources already return ranked results)
            top_datasets = all_datasets[:max_results]
            
            logger.info(f"🎯 Returning {len(top_datasets)} dataset recommendations")
            print(f"🎯 DATASET RESULTS: {len(top_datasets)} recommendations found")
            
            # Format for agent consumption
            return self._format_for_agent(top_datasets, domain)
            
        except Exception as e:
            logger.error(f"❌ Dataset recommendation error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Dataset Recommendation Error\n\n{str(e)}\n\nTry loading your own data with `pd.read_csv('your_data.csv')`"
            }
    
    def _extract_keywords(self, user_query: str) -> List[str]:
        """Extract simple keywords from user query"""
        # Remove common words and extract meaningful terms
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "within", "without", "against", "towards", "upon", "across", "around", "under", "over", "want", "need", "help", "train", "model", "data"}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', user_query.lower())
        keywords = [word for word in words if word not in stop_words]
        return keywords[:10]  # Limit to top 10 keywords
    
    def _format_for_agent(self, datasets: List[Dataset], domain: str) -> Dict[str, Any]:
        """Format dataset recommendations for agent consumption"""
        try:
            if not datasets:
                return {
                    "success": True,
                    "training_result": "## 📊 No Suitable Datasets Found\n\nNo datasets found matching your criteria. Try loading your own data with `pd.read_csv('your_data.csv')`"
                }
            
            # Create formatted recommendation text
            result_text = f"## 📊 Recommended Datasets\n\nBased on your query, here are {len(datasets)} relevant datasets:\n\n"
            
            for i, dataset in enumerate(datasets[:3], 1):  # Show top 3
                loading_code = self.generate_loading_code(dataset, "df")
                
                result_text += f"""### {i}. {dataset.title}
**Source:** {dataset.source.title()} | **Domain:** {dataset.domain.title()}
**Size:** {dataset.rows} rows × {dataset.columns} columns ({dataset.size_mb}MB)

{dataset.description}

**Loading Code:**
```python
{loading_code}
```

---

"""
            
            result_text += "\n**Next Steps:** Choose a dataset above, run the loading code, then retry your ML training request."
            
            return {
                "success": True,
                "training_result": result_text
            }
            
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Formatting Error\n\n{str(e)}"
            }
    
    def generate_loading_code(self, dataset: Dataset, variable_name: str = "df") -> str:
        """Generate code to load a recommended dataset"""
        try:
            if dataset.source == "uci":
                return f"""
# Load {dataset.title} from UCI ML Repository
import pandas as pd

# Note: You may need to check the UCI page for exact file format
{variable_name} = pd.read_csv('{dataset.download_url}')
print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
{variable_name}.head()

# Dataset info: {dataset.url}"""
            
            elif dataset.source == "kaggle":
                return f"""
# Load {dataset.title} from Kaggle
import pandas as pd

# Note: You may need Kaggle API or manual download for some datasets
# For public datasets, try direct download:
{variable_name} = pd.read_csv('{dataset.download_url}')
print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
{variable_name}.head()

# Dataset page: {dataset.url}"""
            
            else:
                return f"""
# Load {dataset.title} dataset
import pandas as pd

{variable_name} = pd.read_csv('{dataset.download_url}')
print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
{variable_name}.head()"""
                
        except Exception as e:
            logger.warning(f"Code generation error: {e}")
            return f"# Error generating loading code for {dataset.title}"


class DatasetSource:
    """Base class for dataset sources"""
    
    def search_datasets(self, keywords: List[str], domain: str, max_results: int = 5) -> List[Dataset]:
        """Search for datasets matching keywords and domain"""
        # Base implementation - subclasses must override
        _ = keywords, domain, max_results  # Suppress unused warnings
        raise NotImplementedError



class UCIMLRepoSource(DatasetSource):
    """Interface to UCI ML Repository with web scraping"""
    
    def __init__(self):
        self.base_url = "https://archive.ics.uci.edu"
        self.datasets_url = f"{self.base_url}/datasets"
        
    def search_datasets(self, keywords: List[str], domain: str, max_results: int = 5) -> List[Dataset]:
        """Search UCI ML Repository datasets by scraping"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            logger.info("🔍 Scraping UCI ML Repository...")
            print("🔍 UCI SCRAPER: Fetching dataset listings")
            
            # Get the main datasets page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(self.datasets_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find dataset cards/containers (new UCI website structure)
            datasets = []
            # Look for dataset containers with various possible structures
            dataset_cards = (
                soup.find_all('div', class_=lambda x: x and 'dataset' in str(x).lower()) or
                soup.find_all('a', href=lambda x: x and '/dataset/' in str(x)) or
                soup.find_all('div', class_=lambda x: x and 'card' in str(x).lower()) or
                soup.select('div[data-testid*="dataset"]') or
                soup.find_all('article')[:50]
            )
            
            logger.info(f"📋 Found {len(dataset_cards)} UCI dataset candidates to process")
            
            # Debug: Show what type of elements we found
            if dataset_cards:
                logger.info(f"📋 Sample card HTML: {str(dataset_cards[0])[:0]}...")
            
            for card in dataset_cards[:50]:  # Limit processing to first 50 for performance
                try:
                    dataset = self._parse_dataset_card(card)
                    if dataset:
                        logger.debug(f"🎯 Parsed dataset: {dataset.title}")
                        if self._matches_criteria(dataset, keywords, domain):
                            # Calculate relevance score for ranking
                            dataset.relevance_score = self._calculate_relevance_score(dataset, keywords, domain)
                            datasets.append(dataset)
                            logger.debug(f"✅ Dataset matches criteria: {dataset.title} (score: {dataset.relevance_score})")
                        else:
                            logger.debug(f"❌ Dataset doesn't match criteria: {dataset.title}")
                        
                        if len(datasets) >= max_results * 2:  # Get more candidates for ranking
                            break
                except Exception as e:
                    logger.debug(f"Error parsing card: {e}")
                    continue
            
            # Sort by relevance score and return top results
            datasets.sort(key=lambda d: getattr(d, 'relevance_score', 0), reverse=True)
            
            logger.info(f"✅ Found {len(datasets)} matching UCI datasets")
            print(f"✅ UCI SCRAPER: Found {len(datasets)} matching datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"❌ UCI scraping error: {e}")
            print(f"❌ UCI SCRAPER ERROR: {e}")
            return []
    
    def _parse_dataset_card(self, card):
        """Parse a dataset card from UCI website"""
        try:
            # Extract dataset name and URL
            name = ""
            dataset_url = ""
            description = ""
            
            # Method 1: Look for dataset link
            dataset_link = card.find('a', href=lambda x: x and '/dataset/' in str(x))
            if dataset_link:
                name = dataset_link.get_text().strip()
                href = dataset_link.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            # Method 2: If card IS the link
            elif card.name == 'a' and '/dataset/' in str(card.get('href', '')):
                name = card.get_text().strip()
                href = card.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            if not name or not dataset_url:
                return None
            
            # Extract description/abstract
            desc_element = (
                card.find('p') or 
                card.find('div', class_=lambda x: x and 'abstract' in str(x).lower()) or
                card.find('div', class_=lambda x: x and 'description' in str(x).lower())
            )
            if desc_element:
                description = desc_element.get_text().strip()
            
            # Extract metadata (instances, features, task type)
            instances = 100  # default
            attributes = 10  # default
            task_text = ""
            data_types_text = "Tabular"
            
            # Look for metadata text
            all_text = card.get_text().lower()
            
            # Extract numbers for instances/features
            import re
            instance_match = re.search(r'(\d+)\s*instances?', all_text)
            if instance_match:
                instances = int(instance_match.group(1))
                
            feature_match = re.search(r'(\d+)\s*features?', all_text)
            if feature_match:
                attributes = int(feature_match.group(1))
            
            # Extract task type
            if 'classification' in all_text:
                task_text = "Classification"
            elif 'regression' in all_text:
                task_text = "Regression"
            elif 'clustering' in all_text:
                task_text = "Clustering"
            else:
                task_text = "Classification"  # default
            
            # Determine domain
            domain = self._determine_domain(data_types_text, task_text, name)
            
            # Create dataset
            dataset = Dataset(
                title=name,
                description=description or f"UCI ML Repository dataset for {task_text.lower()}",
                source="uci",
                domain=domain,
                url=dataset_url,
                download_url=self._construct_download_url(dataset_url, name),
                size_mb=round(instances * attributes * 0.001, 2),  # Rough estimate
                rows=instances,
                columns=attributes,
                tags=self._extract_tags(task_text, data_types_text, name),
                difficulty="beginner" if instances < 1000 else "intermediate"
            )
            
            logger.debug(f"📊 Parsed UCI dataset: {name}")
            return dataset
            
        except Exception as e:
            logger.debug(f"Card parsing error: {e}")
            return None
    
    def _determine_domain(self, data_types: str, task: str, name: str):
        """Determine the domain based on UCI dataset info"""
        text_lower = f"{data_types} {task} {name}".lower()
        
        # Enhanced time series indicators
        timeseries_keywords = [
            'time', 'temporal', 'series', 'forecast', 'sequence', 'sequential',
            'daily', 'monthly', 'yearly', 'hourly', 'stock', 'price', 'trend',
            'seasonal', 'weather', 'climate', 'sensor', 'monitoring', 'tracking'
        ]
        if any(indicator in text_lower for indicator in timeseries_keywords):
            return "timeseries"
        
        # Enhanced multimodal indicators  
        multimodal_keywords = [
            'image', 'text', 'multivariate', 'mixed', 'vision', 'nlp',
            'photo', 'picture', 'language', 'speech', 'audio', 'video'
        ]
        if any(indicator in text_lower for indicator in multimodal_keywords):
            return "multimodal"
        
        # Default to tabular
        return "tabular"
    
    def _extract_tags(self, task: str, data_types: str, name: str):
        """Extract relevant tags from UCI dataset info using comprehensive categories"""
        tags = []
        text_lower = f"{task} {data_types} {name}".lower()
        
        # Enhanced task-based tags (using UCI categories)
        task_indicators = {
            'classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
            'regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'forecast', 'predict value'],
            'clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
            'recommendation': ['recommendation', 'recommender', 'collaborative filtering', 'content-based'],
            'causal-discovery': ['causal', 'causality', 'cause', 'effect', 'causal inference'],
            'feature-selection': ['feature selection', 'variable selection', 'dimensionality']
        }
        
        for tag, keywords in task_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        # Enhanced subject area tags (using UCI categories)
        subject_indicators = {
            'life-sciences': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital', 'drug', 'cancer', 'heart', 'brain'],
            'physical-sciences': ['physics', 'chemistry', 'astronomy', 'energy', 'particle', 'chemical', 'molecular', 'quantum', 'weather', 'climate'],
            'cs-engineering': ['computer', 'software', 'algorithm', 'network', 'system', 'engineering', 'technology', 'robot', 'ai', 'machine learning'],
            'social-sciences': ['social', 'psychology', 'sociology', 'demographic', 'census', 'population', 'survey', 'behavior', 'education', 'student'],
            'business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic', 'profit', 'revenue', 'market'],
            'game': ['game', 'chess', 'poker', 'tic-tac-toe', 'connect', 'puzzle', 'strategy'],
            'law': ['legal', 'law', 'court', 'judge', 'crime', 'criminal', 'justice']
        }
        
        for tag, keywords in subject_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        # Enhanced data type tags (using UCI categories)
        data_type_indicators = {
            'multivariate': ['multivariate', 'multiple variables', 'multi-dimensional', 'several features'],
            'univariate': ['univariate', 'single variable', 'one dimension', 'single feature'],
            'sequential': ['sequential', 'sequence', 'ordered', 'series'],
            'time-series': ['time series', 'temporal', 'time-based', 'chronological', 'timeseries'],
            'text': ['text', 'document', 'corpus', 'natural language', 'nlp', 'linguistic'],
            'images': ['image', 'picture', 'visual', 'pixel', 'photo', 'computer vision'],
            'spatio-temporal': ['spatial', 'geographic', 'location', 'geo', 'coordinates']
        }
        
        for tag, keywords in data_type_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
            
        return tags[:8]  # Limit to 8 tags for better coverage
    
    def _construct_download_url(self, dataset_url: str, name: str):
        """Construct download URL for UCI dataset"""
        # Most UCI datasets have data files in a predictable pattern
        # dataset_url provides context but we construct standardized path
        _ = dataset_url  # Keep parameter for future use
        clean_name = name.lower().replace(' ', '-').replace('/', '-')
        return f"{self.base_url}/machine-learning-databases/{clean_name}/"
    
    def _matches_criteria(self, dataset: Dataset, keywords: List[str], domain: str):
        """Check if dataset matches search criteria"""
        if not keywords:
            return True
            
        # Check domain compatibility (more lenient)
        if domain != "tabular" and dataset.domain != domain and dataset.domain != "tabular":
            return False
            
        # Check keyword matches
        searchable_text = f"{dataset.title} {dataset.description} {' '.join(dataset.tags)}".lower()
        query_text = ' '.join(keywords).lower()
        
        # Enhanced matching using UCI-style categorical matching
        important_matches = False
        
        # Task type matching (from inspiration code)
        task_indicators = {
            'classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
            'regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'forecast', 'predict value'],
            'clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
            'timeseries': ['time series', 'temporal', 'time-based', 'chronological', 'timeseries', 'sequential']
        }
        
        # Subject area matching (from inspiration code)  
        subject_indicators = {
            'life-sciences': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital'],
            'business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic'],
            'physical-sciences': ['physics', 'chemistry', 'astronomy', 'energy', 'weather', 'climate'],
            'social-sciences': ['social', 'psychology', 'demographic', 'census', 'population', 'survey', 'education']
        }
        
        # Data type matching (from inspiration code)
        data_type_indicators = {
            'time-series': ['time series', 'temporal', 'time-based', 'chronological', 'timeseries'],
            'multivariate': ['multivariate', 'multiple variables', 'multi-dimensional'],
            'text': ['text', 'document', 'nlp', 'natural language'],
            'images': ['image', 'picture', 'visual', 'computer vision']
        }
        
        # Check for any important matches
        for category_dict in [task_indicators, subject_indicators, data_type_indicators]:
            for category, keywords_list in category_dict.items():
                if any(kw in query_text for kw in keywords_list):
                    if any(kw in searchable_text for kw in keywords_list) or category in searchable_text:
                        important_matches = True
                        break
            if important_matches:
                break
        
        # General keyword matches
        keyword_matches = 0
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() in searchable_text:
                keyword_matches += 1
        
        # Be more selective - require either important matches or significant keyword overlap
        if important_matches:
            return True
        elif keyword_matches >= 2:  # Need at least 2 keyword matches
            return True
        elif domain == "tabular" and keyword_matches > 0:
            return True
        else:
            return False
    
    def _calculate_relevance_score(self, dataset: Dataset, keywords: List[str], domain: str) -> float:
        """Calculate relevance score for ranking datasets"""
        score = 0.0
        searchable_text = f"{dataset.title} {dataset.description} {' '.join(dataset.tags)}".lower()
        query_text = ' '.join(keywords).lower()
        
        # Domain match bonus
        if dataset.domain == domain:
            score += 10.0
        
        # Enhanced important keyword matching based on UCI categories
        task_keywords = {
            'classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
            'regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'forecast', 'predict value'],
            'clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
            'timeseries': ['time series', 'temporal', 'time-based', 'chronological', 'timeseries', 'sequential', 'stock', 'weather'],
            'multimodal': ['multimodal', 'image', 'text', 'vision', 'nlp', 'computer vision', 'natural language']
        }
        
        subject_keywords = {
            'medical': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital'],
            'business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic'],
            'science': ['physics', 'chemistry', 'astronomy', 'energy', 'particle', 'chemical', 'weather', 'climate'],
            'social': ['social', 'psychology', 'demographic', 'census', 'population', 'survey', 'education']
        }
        
        # Check for task matches (highest weight)
        for task_type, keywords in task_keywords.items():
            if any(kw in query_text for kw in keywords) and any(kw in searchable_text for kw in keywords):
                score += 8.0
                
        # Check for subject area matches (high weight)
        for subject, keywords in subject_keywords.items():
            if any(kw in query_text for kw in keywords) and any(kw in searchable_text for kw in keywords):
                score += 6.0
        
        # Title keyword matches (medium weight)
        title_lower = dataset.title.lower()
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() in title_lower:
                score += 3.0
        
        # Description keyword matches (lower weight)
        desc_lower = dataset.description.lower()
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() in desc_lower:
                score += 1.0
        
        # Tag matches (medium weight)
        for tag in dataset.tags:
            for keyword in keywords:
                if len(keyword) > 3 and keyword.lower() in tag.lower():
                    score += 2.0
        
        return score
