package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func longestMonotonicSubarray(nums []int) int {
	if len(nums) == 1 {
		return 1
	}
	var maxLength int = 1
	var upLength int = 1
	var downLength int = 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			upLength++
			downLength = 1
		}
		if nums[i] < nums[i-1] {
			downLength++
			upLength = 1
		}
		if nums[i] == nums[i-1] {
			downLength = 1
			upLength = 1
		}
		if downLength > maxLength {
			maxLength = downLength
		}
		if upLength > maxLength {
			maxLength = upLength
		}
	}
	return maxLength
}

func singleNumber(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	var xorElement = nums[0]
	for i := 1; i < len(nums); i++ {
		xorElement = xorElement ^ nums[i]
	}
	return xorElement
}

func maxSubarraySumCircular(nums []int) int {
	var maxSum = nums[0]
	var minSum = nums[0]
	var tempMax = 0
	var tempMin = 0
	var totalSum = 0

	for i := 0; i < len(nums); i++ {
		tempMax = max(tempMax, 0) + nums[i]
		maxSum = max(tempMax, maxSum)

		tempMin = min(tempMin, 0) + nums[i]
		minSum = min(tempMin, minSum)

		totalSum += nums[i]
	}

	if minSum == totalSum {
		return maxSum
	} else {
		return max(maxSum, totalSum-minSum)
	}
}

func candy(ratings []int) int {
	var n = len(ratings)
	var totalCandies = n
	for i := 1; i < n; {
		if ratings[i] == ratings[i-1] {
			i++
			continue
		}
		var currentPeek = 0
		for i < n && ratings[i] > ratings[i-1] {
			currentPeek++
			totalCandies += currentPeek
			i++
		}
		if i == n {
			return totalCandies
		}
		var currentValley = 0
		for i < n && ratings[i] < ratings[i-1] {
			currentValley++
			totalCandies += currentValley
			i++
		}
		totalCandies -= min(currentValley, currentPeek)
	}
	return totalCandies
}

func coinChange(coins []int, amount int) int {
	var dp = make([]int, amount+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i < len(dp); i++ {
		for j := 0; j < len(coins); j++ {
			if i-coins[j] >= 0 {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	return dp[amount]
}

var happenedNumber []int

func getTotalSquare(number int) int {
	var res int = 0
	if number == 1 {
		return 1
	}
	for number > 0 {
		res += (number % 10) * (number % 10)
		number /= 10
	}
	if slices.Contains(happenedNumber, res) {
		return -1
	}
	happenedNumber = append(happenedNumber, res)
	return getTotalSquare(res)
}

func isHappy(n int) bool {
	return getTotalSquare(n) == 1
}

func maxAscendingSum(nums []int) int {
	var maxSum int = nums[0]
	var tempSum int = nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			tempSum += nums[i]
		} else {
			if tempSum > maxSum {
				maxSum = tempSum
			}
			tempSum = nums[i]
		}
		if tempSum > maxSum {
			maxSum = tempSum
		}
	}
	return maxSum
}
func areAlmostEqual(s1 string, s2 string) bool {
	var diffCount = 0
	var diffColls1 = make([]byte, 0)
	var diffColls2 = make([]byte, 0)
	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			diffCount++
			diffColls2 = append(diffColls2, s2[i])
			diffColls1 = append(diffColls1, s1[i])
		}
		if diffCount > 2 {
			return false
		}
	}
	if diffCount == 2 {
		return diffColls1[0] == diffColls2[1] && diffColls1[1] == diffColls2[0]
	}
	if diffCount != 0 {
		return false
	} else {
		return true
	}
}

func tupleSameProduct(nums []int) int {
	var freqMap = make(map[int]int)
	for i := 0; i < len(nums)-1; i++ {
		for j := i + 1; j < len(nums); j++ {
			if freqMap[nums[i]*nums[j]] == 0 {
				freqMap[nums[i]*nums[j]] = 1
			} else {
				freqMap[nums[i]*nums[j]]++
			}
		}
	}
	var count int = 0
	for _, value := range freqMap {
		if value > 1 {
			count += 4 * value * (value - 1)
		}
	}
	return count
}

func queryResults(limit int, queries [][]int) []int {
	var result = make([]int, len(queries))
	var freqMap = make(map[int]int)
	var coloredBall = make(map[int]int)
	var diffColorCount = 0
	for i := 0; i < len(queries); i++ {
		if coloredBall[queries[i][0]] == 0 {
			if freqMap[queries[i][1]] == 0 {
				diffColorCount++
			}
		} else {
			var oldColor = coloredBall[queries[i][0]]
			var newColor = queries[i][1]
			freqMap[oldColor]--
			if freqMap[oldColor] == 0 {
				diffColorCount--
			}
			if freqMap[newColor] == 0 {
				diffColorCount++
			}
		}
		coloredBall[queries[i][0]] = queries[i][1]
		freqMap[queries[i][1]]++
		result[i] = diffColorCount
	}
	return result
}

type NumberContainers struct {
	numbers map[int]int
	index   map[int][]int
}

func Constructor() NumberContainers {
	return NumberContainers{
		numbers: make(map[int]int),
		index:   make(map[int][]int),
	}
}

func (this *NumberContainers) Change(index int, number int) {
	var oldValue = this.numbers[index]
	this.numbers[index] = number
	if oldValue != 0 {
		this.index[oldValue] = DeleteSorted(this.index[oldValue], index)
	}
	this.index[number] = InsertSorted(this.index[number], index)
}

func (this *NumberContainers) Find(number int) int {
	if this.index[number] != nil {
		return -1
	} else {
		return this.index[number][0]
	}
}

func InsertSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	arr = append(arr, 0)
	copy(arr[idx+1:], arr[idx:])
	arr[idx] = val
	return arr
}

func DeleteSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	if idx < len(arr) && arr[idx] == val {
		arr = append(arr[:idx], arr[idx+1:]...)
	}
	return arr
}

func countBadPairs(nums []int) int64 {
	var notBadPairCount = 0
	var mapper = make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mapper[nums[i]-i]++
		notBadPairCount += mapper[nums[i]-i]
	}

	return int64(len(nums)*(len(nums)-1)/2 - notBadPairCount)
}

type stack[T any] struct {
	Push   func(T)
	Pop    func() T
	Length func() int
}

func Stack[T any]() stack[T] {
	slice := make([]T, 0)
	return stack[T]{
		Push: func(i T) {
			slice = append(slice, i)
		},
		Pop: func() T {
			res := slice[len(slice)-1]
			slice = slice[:len(slice)-1]
			return res
		},
		Length: func() int {
			return len(slice)
		},
	}
}
func solve(index int, digits string, comb []string, ans *[]string, temp string) {
	if index == len(digits) {
		*ans = append(*ans, temp)
		return
	}

	for _, ch := range comb[digits[index]-'0'] {
		solve(index+1, digits, comb, ans, temp+string(ch))
	}
}

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	comb := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var ans []string
	solve(0, digits, comb, &ans, "")
	return ans
}
func zeroFilledSubarray(nums []int) int64 {
	result := 0
	isValid := false
	startIndex := -1
	endIndex := -1
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			if isValid {
				endIndex = i
			} else {
				isValid = true
				startIndex = i
				endIndex = i
			}
			if i == len(nums)-1 {
				n := endIndex - startIndex + 1
				result += n + (n-1)*n/2
			}
		} else if nums[i] != 0 && isValid {
			n := endIndex - startIndex + 1
			result += n + (n-1)*n/2
			isValid = false
		}
	}
	return int64(result)
}

func repeatedCharacter(s string) byte {
	vector := 0

	for _, c := range s {
		if vector&(1<<(c-'a')) > 0 {
			return byte(c)
		}
		vector |= 1 << (c - 'a')
	}
	return ' '
}

func equalPairs(grid [][]int) int {
	var result int = 0
	hashedRow := make(map[string]int)
	for i := 0; i < len(grid); i++ {
		var newRow string = ""
		for j := 0; j < len(grid[i]); j++ {
			newRow += string(grid[i][j] * 10)
		}
		hashedRow[newRow]++
	}
	for i := 0; i < len(grid[0]); i++ {
		var newRow string = ""
		for j := 0; j < len(grid); j++ {
			newRow += string(grid[j][i] * 10)
		}
		if hashedRow[newRow] > 0 {
			result += hashedRow[newRow]
		}
	}
	return result
}

// MLE
func checkAround(i int, j int, matrix [][]int) int64 {
	return int64(matrix[i][j] + matrix[i+1][j] + matrix[i][j+1] + matrix[i+1][j+1])
}

func failcountBlackBlocks(m int, n int, coordinates [][]int) []int64 {
	result := make([]int64, 5)
	matrix := make([][]int, m)
	for i := 0; i < m; i++ {
		matrix[i] = make([]int, n)
	}
	for _, value := range coordinates {
		matrix[value[0]][value[1]] = 1
	}
	for i := 0; i < m-1; i++ {
		for j := 0; j < n-1; j++ {
			result[checkAround(i, j, matrix)]++
		}
	}
	return result
}

func getHappyString(n int, k int) string {
	stringStack := []string{""}
	index := 0
	for len(stringStack) > 0 {
		current := stringStack[len(stringStack)-1]
		stringStack = stringStack[:len(stringStack)-1]
		if len(current) == n {
			index++
			if index == k {
				return current
			}
			continue
		}
		for i := 'c'; i >= 'a'; i-- {
			if len(current) == 0 || current[len(current)-1] != uint8(i) {
				stringStack = append(stringStack, current+string(i))
			}
		}
	}
	return ""
}

func maximumsSplicedArray(nums1 []int, nums2 []int) int {
	sum1 := 0
	sum2 := 0
	maxProfit := 0
	maxProfit2 := 0
	currentSum := 0
	currentSum2 := 0
	for i := 0; i < len(nums1); i++ {
		sum1 += nums1[i]
		sum2 += nums2[i]
		temp := nums2[i] - nums1[i]
		temp2 := nums1[i] - nums2[i]
		currentSum = int(math.Max(float64(currentSum)+float64(temp), float64(temp)))
		maxProfit = int(math.Max(float64(maxProfit), float64(currentSum)))
		currentSum2 = int(math.Max(float64(currentSum2)+float64(temp2), float64(temp2)))
		maxProfit2 = int(math.Max(float64(maxProfit2), float64(currentSum2)))
	}
	return int(math.Max(math.Max(float64(sum1+maxProfit), float64(sum1)), math.Max(float64(sum2+maxProfit2), float64(sum2))))
}

func poorPigs(buckets int, minutesToDie int, minutesToTest int) int {
	result := 0
	for math.Pow(float64((minutesToTest/minutesToDie)+1), float64(result)) < float64(buckets) {
		result++
	}
	return result
}

func main() {
	a1 := []int{10, 20, 50, 15, 30, 10}
	a2 := []int{40, 20, 10, 100, 10, 10}
	fmt.Print(maximumsSplicedArray(a1, a2))
}
