# Laravel Integration Guide for CropTAP API

This guide shows how to integrate the CropTAP Crop Recommendation API into your Laravel application.

## Prerequisites

- Laravel 9.x or higher
- PHP 8.0 or higher
- CropTAP API running on `http://127.0.0.1:5000`

## Table of Contents

1. [Quick Start](#quick-start)
2. [Service Class Implementation](#service-class-implementation)
3. [Controller Example](#controller-example)
4. [Request Validation](#request-validation)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)

---

## Quick Start

### 1. Install HTTP Client (Optional)

Laravel includes Guzzle by default, but you can also use Laravel's HTTP facade:

```bash
# No installation needed - Laravel HTTP facade is built-in
```

### 2. Configure API URL

Add to your `.env` file:

```env
CROPTAP_API_URL=http://127.0.0.1:5000
CROPTAP_API_TIMEOUT=30
```

Add to `config/services.php`:

```php
return [
    // ... other services
    
    'croptap' => [
        'url' => env('CROPTAP_API_URL', 'http://127.0.0.1:5000'),
        'timeout' => env('CROPTAP_API_TIMEOUT', 30),
    ],
];
```

---

## Service Class Implementation

Create a service class to handle API communication:

### Create Service File

```bash
php artisan make:class Services/CropTapService
```

### `app/Services/CropTapService.php`

```php
<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Exception;

class CropTapService
{
    protected string $baseUrl;
    protected int $timeout;

    public function __construct()
    {
        $this->baseUrl = config('services.croptap.url');
        $this->timeout = config('services.croptap.timeout');
    }

    /**
     * Get crop recommendations based on farmer input
     *
     * @param array $data
     * @return array
     * @throws Exception
     */
    public function getRecommendations(array $data): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post("{$this->baseUrl}/recommend", [
                    'province' => $data['province'],
                    'municipality' => $data['municipality'],
                    'nitrogen' => [
                        'value' => $data['nitrogen']
                    ],
                    'phosphorus' => [
                        'value' => $data['phosphorus']
                    ],
                    'potassium' => [
                        'value' => $data['potassium']
                    ],
                    'ph_min' => $data['ph_min'],
                    'ph_max' => $data['ph_max'],
                    'soil_type' => $data['soil_type']
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            // Handle API errors
            $error = $response->json();
            throw new Exception(
                $error['detail'] ?? 'API request failed',
                $response->status()
            );

        } catch (Exception $e) {
            Log::error('CropTAP API Error: ' . $e->getMessage(), [
                'data' => $data,
                'trace' => $e->getTraceAsString()
            ]);
            throw $e;
        }
    }

    /**
     * Check API health status
     *
     * @return array
     */
    public function healthCheck(): array
    {
        try {
            $response = Http::timeout(5)
                ->get("{$this->baseUrl}/health");

            return $response->successful() 
                ? $response->json() 
                : ['status' => 'unhealthy'];

        } catch (Exception $e) {
            return ['status' => 'unreachable', 'error' => $e->getMessage()];
        }
    }

    /**
     * Get API information
     *
     * @return array
     */
    public function apiInfo(): array
    {
        try {
            $response = Http::timeout(5)
                ->get($this->baseUrl);

            return $response->successful() ? $response->json() : [];

        } catch (Exception $e) {
            return ['error' => $e->getMessage()];
        }
    }
}
```

---

## Controller Example

### Create Controller

```bash
php artisan make:controller CropRecommendationController
```

### `app/Http/Controllers/CropRecommendationController.php`

```php
<?php

namespace App\Http\Controllers;

use App\Services\CropTapService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Exception;

class CropRecommendationController extends Controller
{
    protected CropTapService $cropTapService;

    public function __construct(CropTapService $cropTapService)
    {
        $this->cropTapService = $cropTapService;
    }

    /**
     * Get crop recommendations
     */
    public function getRecommendations(Request $request): JsonResponse
    {
        // Validate request
        $validated = $request->validate([
            'province' => 'required|string',
            'municipality' => 'required|string',
            'nitrogen' => 'required|in:Low,Medium,High',
            'phosphorus' => 'required|in:Low,Medium,High',
            'potassium' => 'required|in:Low,Medium,High',
            'ph_min' => 'required|numeric|min:0|max:14',
            'ph_max' => 'required|numeric|min:0|max:14|gte:ph_min',
            'soil_type' => 'required|string',
        ]);

        try {
            $recommendations = $this->cropTapService->getRecommendations($validated);

            return response()->json([
                'success' => true,
                'data' => $recommendations
            ]);

        } catch (Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Failed to get recommendations',
                'error' => $e->getMessage()
            ], $e->getCode() ?: 500);
        }
    }

    /**
     * Display recommendation form (for web routes)
     */
    public function showForm()
    {
        return view('recommendations.form');
    }

    /**
     * Display recommendations (for web routes)
     */
    public function showRecommendations(Request $request)
    {
        $validated = $request->validate([
            'province' => 'required|string',
            'municipality' => 'required|string',
            'nitrogen' => 'required|in:Low,Medium,High',
            'phosphorus' => 'required|in:Low,Medium,High',
            'potassium' => 'required|in:Low,Medium,High',
            'ph_min' => 'required|numeric|min:0|max:14',
            'ph_max' => 'required|numeric|min:0|max:14|gte:ph_min',
            'soil_type' => 'required|string',
        ]);

        try {
            $recommendations = $this->cropTapService->getRecommendations($validated);

            return view('recommendations.results', [
                'recommendations' => $recommendations,
                'input' => $validated
            ]);

        } catch (Exception $e) {
            return back()
                ->withInput()
                ->withErrors(['error' => $e->getMessage()]);
        }
    }

    /**
     * Check API health
     */
    public function healthCheck(): JsonResponse
    {
        $health = $this->cropTapService->healthCheck();

        return response()->json($health);
    }
}
```

---

## Request Validation

### Create Form Request

```bash
php artisan make:request CropRecommendationRequest
```

### `app/Http/Requests/CropRecommendationRequest.php`

```php
<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class CropRecommendationRequest extends FormRequest
{
    public function authorize(): bool
    {
        return true;
    }

    public function rules(): array
    {
        return [
            'province' => 'required|string|max:255',
            'municipality' => 'required|string|max:255',
            'nitrogen' => 'required|in:Low,Medium,High',
            'phosphorus' => 'required|in:Low,Medium,High',
            'potassium' => 'required|in:Low,Medium,High',
            'ph_min' => 'required|numeric|min:0|max:14',
            'ph_max' => 'required|numeric|min:0|max:14|gte:ph_min',
            'soil_type' => 'required|string|max:255',
        ];
    }

    public function messages(): array
    {
        return [
            'province.required' => 'Province is required',
            'municipality.required' => 'Municipality is required',
            'nitrogen.in' => 'Nitrogen must be Low, Medium, or High',
            'phosphorus.in' => 'Phosphorus must be Low, Medium, or High',
            'potassium.in' => 'Potassium must be Low, Medium, or High',
            'ph_min.required' => 'Minimum pH is required',
            'ph_max.required' => 'Maximum pH is required',
            'ph_max.gte' => 'Maximum pH must be greater than or equal to minimum pH',
            'soil_type.required' => 'Soil type is required',
        ];
    }
}
```

---

## Routes

### `routes/api.php`

```php
use App\Http\Controllers\CropRecommendationController;

Route::prefix('crop-recommendations')->group(function () {
    Route::post('/', [CropRecommendationController::class, 'getRecommendations']);
    Route::get('/health', [CropRecommendationController::class, 'healthCheck']);
});
```

### `routes/web.php`

```php
use App\Http\Controllers\CropRecommendationController;

Route::prefix('recommendations')->group(function () {
    Route::get('/', [CropRecommendationController::class, 'showForm'])
        ->name('recommendations.form');
    Route::post('/', [CropRecommendationController::class, 'showRecommendations'])
        ->name('recommendations.show');
});
```

---

## Usage Examples

### Example 1: API Request (JSON)

```bash
POST http://your-laravel-app.com/api/crop-recommendations
Content-Type: application/json

{
    "province": "Laguna",
    "municipality": "Los Baños",
    "nitrogen": "Medium",
    "phosphorus": "High",
    "potassium": "Medium",
    "ph_min": 6.0,
    "ph_max": 7.0,
    "soil_type": "Clay Loam"
}
```

### Example 2: Using in Another Service

```php
use App\Services\CropTapService;

class FarmManagementService
{
    public function __construct(
        protected CropTapService $cropTapService
    ) {}

    public function analyzeFarm(Farm $farm)
    {
        $recommendations = $this->cropTapService->getRecommendations([
            'province' => $farm->province,
            'municipality' => $farm->municipality,
            'nitrogen' => $farm->soil_nitrogen_level,
            'phosphorus' => $farm->soil_phosphorus_level,
            'potassium' => $farm->soil_potassium_level,
            'ph_min' => $farm->soil_ph_min,
            'ph_max' => $farm->soil_ph_max,
            'soil_type' => $farm->soil_type,
        ]);

        // Process recommendations
        return $recommendations;
    }
}
```

### Example 3: Blade View Form

```blade
<!-- resources/views/recommendations/form.blade.php -->
<form method="POST" action="{{ route('recommendations.show') }}">
    @csrf
    
    <div class="form-group">
        <label>Province</label>
        <input type="text" name="province" class="form-control" required>
    </div>

    <div class="form-group">
        <label>Municipality</label>
        <input type="text" name="municipality" class="form-control" required>
    </div>

    <div class="form-group">
        <label>Nitrogen Level</label>
        <select name="nitrogen" class="form-control" required>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
        </select>
    </div>

    <div class="form-group">
        <label>Phosphorus Level</label>
        <select name="phosphorus" class="form-control" required>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
        </select>
    </div>

    <div class="form-group">
        <label>Potassium Level</label>
        <select name="potassium" class="form-control" required>
            <option value="Low">Low</option>
            <option value="Medium">Medium</option>
            <option value="High">High</option>
        </select>
    </div>

    <div class="row">
        <div class="col-md-6">
            <div class="form-group">
                <label>Minimum pH</label>
                <input type="number" name="ph_min" step="0.1" min="0" max="14" class="form-control" required>
            </div>
        </div>
        <div class="col-md-6">
            <div class="form-group">
                <label>Maximum pH</label>
                <input type="number" name="ph_max" step="0.1" min="0" max="14" class="form-control" required>
            </div>
        </div>
    </div>

    <div class="form-group">
        <label>Soil Type</label>
        <input type="text" name="soil_type" class="form-control" required>
    </div>

    <button type="submit" class="btn btn-primary">Get Recommendations</button>
</form>
```

### Example 4: Display Results

```blade
<!-- resources/views/recommendations/results.blade.php -->
<div class="recommendations">
    <h2>Crop Recommendations for {{ $recommendations['location']['municipality'] }}, {{ $recommendations['location']['province'] }}</h2>

    <div class="climate-summary">
        <h3>Climate Summary</h3>
        <p>Temperature: {{ $recommendations['climate_summary']['avg_temperature'] }}°C</p>
        <p>Rainfall: {{ $recommendations['climate_summary']['avg_rainfall'] }}mm</p>
        <p>Humidity: {{ $recommendations['climate_summary']['avg_humidity'] }}%</p>
    </div>

    <div class="crop-list">
        <h3>Top Recommended Crops</h3>
        @foreach($recommendations['recommendations'] as $crop)
            <div class="crop-card">
                <h4>
                    #{{ $crop['rank'] }} - {{ $crop['crop_name'] }}
                    @if($crop['badge'])
                        <span class="badge">{{ $crop['badge'] }}</span>
                    @endif
                </h4>
                <p><strong>Category:</strong> {{ $crop['category'] }}</p>
                <p><strong>Hybrid Score:</strong> {{ $crop['hybrid_score'] }}/100</p>
                <p><strong>Confidence:</strong> {{ $crop['confidence'] }}%</p>
                <p><strong>Expected Yield:</strong> {{ $crop['expected_yield'] }}</p>
                <p><strong>Planting Season:</strong> {{ $crop['planting_season'] }}</p>
                <p><strong>Days to Harvest:</strong> {{ $crop['days_to_harvest'] }}</p>
                
                @if(count($crop['risks']) > 0)
                    <div class="risks">
                        <strong>Risks:</strong>
                        <ul>
                            @foreach($crop['risks'] as $risk)
                                <li>{{ $risk }}</li>
                            @endforeach
                        </ul>
                    </div>
                @endif

                <p><strong>Why Recommended:</strong> {{ $crop['why_recommended'] }}</p>
                <p><strong>Fertilizer:</strong> {{ $crop['fertilizer_recommendation'] }}</p>
            </div>
        @endforeach
    </div>
</div>
```

---

## Error Handling

### Global Exception Handler

Add to `app/Exceptions/Handler.php`:

```php
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\RequestException;

public function register()
{
    $this->renderable(function (ConnectionException $e, $request) {
        if ($request->expectsJson()) {
            return response()->json([
                'success' => false,
                'message' => 'Unable to connect to CropTAP API',
                'error' => 'Service unavailable'
            ], 503);
        }
        
        return back()->withErrors(['error' => 'Unable to connect to recommendation service']);
    });

    $this->renderable(function (RequestException $e, $request) {
        if ($request->expectsJson()) {
            return response()->json([
                'success' => false,
                'message' => 'API request failed',
                'error' => $e->getMessage()
            ], $e->response->status());
        }
        
        return back()->withErrors(['error' => 'Request to recommendation service failed']);
    });
}
```

---

## Testing

### Feature Test Example

```php
<?php

namespace Tests\Feature;

use Tests\TestCase;
use Illuminate\Support\Facades\Http;

class CropRecommendationTest extends TestCase
{
    public function test_can_get_recommendations()
    {
        Http::fake([
            '*/recommend' => Http::response([
                'location' => [
                    'province' => 'Laguna',
                    'municipality' => 'Los Baños'
                ],
                'recommendations' => [
                    [
                        'crop_name' => 'Rice',
                        'hybrid_score' => 85.5,
                        'rank' => 1
                    ]
                ]
            ], 200)
        ]);

        $response = $this->postJson('/api/crop-recommendations', [
            'province' => 'Laguna',
            'municipality' => 'Los Baños',
            'nitrogen' => 'Medium',
            'phosphorus' => 'High',
            'potassium' => 'Medium',
            'ph_min' => 6.0,
            'ph_max' => 7.0,
            'soil_type' => 'Clay Loam'
        ]);

        $response->assertStatus(200)
            ->assertJsonStructure([
                'success',
                'data' => [
                    'location',
                    'recommendations'
                ]
            ]);
    }
}
```

---

## Performance Tips

1. **Cache Recommendations**: Cache results for common queries
```php
$cacheKey = md5(json_encode($validated));
$recommendations = Cache::remember($cacheKey, 3600, function () use ($validated) {
    return $this->cropTapService->getRecommendations($validated);
});
```

2. **Queue Long-Running Requests**: Use Laravel queues for batch processing
```php
dispatch(new GenerateRecommendationsJob($farmData));
```

3. **Monitor API Health**: Set up scheduled health checks
```php
// In app/Console/Kernel.php
$schedule->call(function () {
    $health = app(CropTapService::class)->healthCheck();
    if ($health['status'] !== 'healthy') {
        // Send alert
    }
})->everyFiveMinutes();
```

---

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure CropTAP API is running on the correct port
   - Check firewall settings
   - Verify `CROPTAP_API_URL` in `.env`

2. **Timeout Errors**
   - Increase timeout in config
   - Check API performance
   - Consider async processing

3. **Validation Errors**
   - Ensure data format matches API requirements
   - Check NPK values are "Low", "Medium", or "High"
   - Verify pH range is 0-14

---

## Additional Resources

- [Laravel HTTP Client Documentation](https://laravel.com/docs/http-client)
- [CropTAP API Documentation](http://127.0.0.1:5000/docs)
- [Laravel Validation](https://laravel.com/docs/validation)
