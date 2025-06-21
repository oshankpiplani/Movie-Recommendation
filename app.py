import os
import ast
from flask import Flask, request, jsonify
from supabase import create_client
from dotenv import load_dotenv
import numpy as np
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import time

load_dotenv()

app = Flask(__name__)

# Configure CORS for production
CORS(app, origins=[
    "https://your-frontend-domain.vercel.app",  # Replace with your frontend URL
    "http://localhost:3000",  # For local development
    "http://127.0.0.1:3000"
])

# Production logging
if os.getenv('RAILWAY_ENVIRONMENT'):
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Load model with error handling
try:
    model = SentenceTransformer("all-mpnet-base-v2")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    model = None

def get_embedding(text: str):
    if model is None:
        raise Exception("Model not loaded")
    return model.encode(text).tolist()

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)

@lru_cache(maxsize=1000)
def cached_embedding(text: str):
    return tuple(get_embedding(text))  

def get_multiple_recommendations(user_input: str, count: int = 3, exclude_titles: list = None):
    if exclude_titles is None:
        exclude_titles = []
    
    input_embedding = list(cached_embedding(user_input))
    
    try:
        search_result = supabase.rpc('match_movies', {
            'query_embedding': input_embedding,
            'match_threshold': 0.01,  
            'match_count': count * 3  
        }).execute()
        
        if search_result.data and len(search_result.data) > 0:
            filtered_results = [
                movie for movie in search_result.data 
                if movie['title'].lower() not in [title.lower() for title in exclude_titles]
            ]
            logger.info(f"Vector search found {len(search_result.data)} results, {len(filtered_results)} after filtering")
            return filtered_results[:count]  
            
    except Exception as e:
        logger.warning(f"Vector search failed: {e}")
    
    logger.info("Using manual search for multiple recommendations")
    try:
        response = supabase.table("movies").select("id,title,overview,embedding,vote_average,release_date").limit(2000).execute()
        movies = response.data
    except Exception as e:
        logger.error(f"Database fetch error: {e}")
        return []

    if not movies:
        return []

    movie_scores = []
    for movie in movies:
        if movie['title'].lower() in [title.lower() for title in exclude_titles]:
            continue
            
        emb = movie.get('embedding')
        if emb:
            if isinstance(emb, str):
                try:
                    emb = ast.literal_eval(emb)
                except (ValueError, SyntaxError):
                    continue
            
            try:
                similarity = cosine_similarity(input_embedding, emb)
                movie_scores.append({
                    **movie,
                    'similarity': similarity
                })
            except Exception as e:
                continue
    movie_scores.sort(key=lambda x: x['similarity'], reverse=True)
    return movie_scores[:count]

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    
    data = request.json
    user_input = data.get('input', '').strip()
    return_multiple = data.get('multiple', False)
    count = min(data.get('count', 3), 5) if return_multiple else 1

    if not user_input:
        return jsonify({
            "reply": "Please provide input text describing what kind of movie you'd like.", 
            "movies": [],
            "processing_time": 0
        })

    logger.info(f"Processing query: '{user_input}' (multiple: {return_multiple})")

    if return_multiple:
        try:
            exclude_titles = []
            if "movies like" in user_input.lower() or "similar to" in user_input.lower():
                words = user_input.lower().split()
                if "like" in words:
                    like_index = words.index("like")
                    if like_index + 1 < len(words):
                        movie_name = " ".join(words[like_index + 1:])
                        exclude_titles.append(movie_name)
            
            recommendations = get_multiple_recommendations(user_input, count, exclude_titles)
            processing_time = time.time() - start_time
            
            if recommendations:
                reply = f"🎯 Here are {len(recommendations)} recommendations for '{user_input}':\n\n"
                
                movies_list = []
                for i, movie in enumerate(recommendations, 1):
                    title = movie['title']
                    overview = movie['overview']
                    rating = movie.get('vote_average', 'N/A')
                    year = movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown'
                    similarity = movie.get('similarity', 0)
                    
                    reply += f"{i}. **{title}** ({year}) - {rating}/10\n"
                    reply += f"   {overview[:100]}{'...' if len(overview) > 100 else ''}\n"
                    reply += f"   Match: {similarity:.1%}\n\n"
                    
                    movies_list.append({
                        "title": title,
                        "overview": overview,
                        "rating": rating,
                        "year": year,
                        "similarity": similarity
                    })
                
                return jsonify({
                    "reply": reply,
                    "movies": movies_list,
                    "count": len(recommendations),
                    "processing_time": processing_time
                })
            else:
                return jsonify({
                    "reply": "Sorry, I couldn't find good movie matches for your request. Try describing what genre, mood, or themes you're interested in!",
                    "movies": [],
                    "processing_time": processing_time
                })
                
        except Exception as e:
            logger.error(f"Multiple recommendations error: {e}")
            return jsonify({
                "reply": f"Error processing your request: {str(e)}",
                "movies": [],
                "processing_time": time.time() - start_time
            })
    
    try:
        input_embedding = list(cached_embedding(user_input))
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return jsonify({
            "reply": f"Error processing your request: {str(e)}", 
            "movie": "",
            "processing_time": time.time() - start_time
        })
    
    try:
        search_result = supabase.rpc('match_movies', {
            'query_embedding': input_embedding,
            'match_threshold': 0.1,
            'match_count': 5
        }).execute()
        
        if search_result.data:
            best_movie = search_result.data[0]
            best_score = best_movie.get('similarity', 0)
            logger.info(f"Vector search found {len(search_result.data)} results")
        else:
            raise Exception("No vector search results")
            
    except Exception as e:
        logger.warning(f"Vector search failed, using manual search: {e}")
        
        try:
            response = supabase.table("movies").select("id,title,overview,embedding,vote_average,release_date").limit(1000).execute()
            movies = response.data
        except Exception as e:
            logger.error(f"Database fetch error: {e}")
            return jsonify({
                "reply": f"Error fetching movies from database: {str(e)}", 
                "movie": "",
                "processing_time": time.time() - start_time
            })

        if not movies:
            return jsonify({
                "reply": "No movies found in the database. Please upload some movies first.", 
                "movie": "",
                "processing_time": time.time() - start_time
            })

        best_score = -1
        best_movie = None
        
        for movie in movies:
            emb = movie.get('embedding')
            if emb:
                if isinstance(emb, str):
                    try:
                        emb = ast.literal_eval(emb)
                    except (ValueError, SyntaxError):
                        continue
                
                try:
                    similarity = cosine_similarity(input_embedding, emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_movie = movie
                except Exception as e:
                    continue

    processing_time = time.time() - start_time
    if best_movie and best_score > 0.1:
        title = best_movie['title']
        overview = best_movie['overview']
        rating = best_movie.get('vote_average', 'N/A')
        year = best_movie.get('release_date', '')[:4] if best_movie.get('release_date') else 'Unknown'
        
        reply = f"🎬 I recommend '{title}' ({year})"
        if rating != 'N/A':
            reply += f" - Rating: {rating}/10"
        reply += f"\n\n📖 Overview: {overview}"
        reply += f"\n\n🎯 Match confidence: {best_score:.1%}"
        
        logger.info(f"Recommended '{title}' with confidence {best_score:.3f} in {processing_time:.3f}s")
        
        return jsonify({
            "reply": reply, 
            "movie": title,
            "confidence": best_score,
            "processing_time": processing_time,
            "details": {
                "title": title,
                "overview": overview,
                "rating": rating,
                "year": year
            }
        })
    else:
        logger.info(f"No good match found. Best score: {best_score}")
        return jsonify({
            "reply": "Sorry, I couldn't find a good movie match for your request. Try describing what genre, mood, or themes you're interested in!", 
            "movie": "",
            "confidence": best_score if best_movie else 0,
            "processing_time": processing_time
        })

@app.route('/search', methods=['POST'])
def search_multiple():
    """Return multiple movie recommendations with improved fallback"""
    data = request.json
    user_input = data.get('input', '').strip()
    count = min(data.get('count', 5), 10)

    if not user_input:
        return jsonify({
            "error": "Please provide input text",
            "results": []
        })

    try:
        exclude_titles = []
        if "movies like" in user_input.lower() or "similar to" in user_input.lower():
            words = user_input.lower().split()
            if "like" in words:
                like_index = words.index("like")
                if like_index + 1 < len(words):
                    movie_name = " ".join(words[like_index + 1:])
                    exclude_titles.append(movie_name)
        
        recommendations = get_multiple_recommendations(user_input, count, exclude_titles)
        
        results = []
        for movie in recommendations:
            year = movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown'
            results.append({
                "title": movie['title'],
                "overview": movie['overview'],
                "rating": movie.get('vote_average', 'N/A'),
                "year": year,
                "similarity": movie.get('similarity', 0)
            })
        
        return jsonify({
            "query": user_input,
            "count": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            "error": str(e),
            "results": []
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "all-mpnet-base-v2" if model else "not loaded",
        "cache_size": cached_embedding.cache_info().currsize if hasattr(cached_embedding, 'cache_info') else 0
    })

@app.route('/stats', methods=['GET'])
def stats():
    """Get database statistics"""
    try:
        response = supabase.table("movies").select("id", count="exact").execute()
        movie_count = response.count
        
        sample_response = supabase.table("movies").select("title,vote_average,release_date").limit(5).execute()
        sample_movies = sample_response.data
        
        return jsonify({
            "total_movies": movie_count,
            "model": "all-mpnet-base-v2" if model else "not loaded",
            "embedding_dimension": "768",
            "sample_movies": sample_movies,
            "cache_info": cached_embedding.cache_info()._asdict() if hasattr(cached_embedding, 'cache_info') else {}
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)})

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the embedding cache"""
    cached_embedding.cache_clear()
    return jsonify({"message": "Cache cleared successfully"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting Movie Recommendation API...")
    print("🤖 Using model: all-mpnet-base-v2")
    print("🗄️ Checking database connection...")
    
    try:
        stats_response = supabase.table("movies").select("id", count="exact").execute()
        print(f"✅ Database connected! Found {stats_response.count} movies")
    except Exception as e:
        print(f"⚠️ Database connection issue: {e}")
    
    port = int(os.environ.get("PORT", 5000))
    debug = not os.getenv('RAILWAY_ENVIRONMENT')
    
    print(f"🌐 Server running on port {port}")
    print("\nAvailable endpoints:")
    print("  POST /query - Get single movie recommendation")
    print("  POST /search - Get multiple movie recommendations")
    print("  GET /health - Health check")
    print("  GET /stats - Database statistics")
    print("  POST /clear-cache - Clear embedding cache")
    
    app.run(host='0.0.0.0', port=port, debug=debug)